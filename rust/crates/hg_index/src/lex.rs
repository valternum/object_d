//! Streaming reader for the Turtle subset used by YAGO 4.6 files.
//!
//! Data files hold one statement per line (`s p o .`, possibly with `;` / `,`
//! abbreviations, or an RDF-star annotation `<< s p o >> mp mo .`). Terms are
//! returned as borrowed slices of the line; escapes are kept verbatim, so equal
//! terms always have equal byte strings (dictionary keys). The same [`Lexer`]
//! serves the multi-line schema parser (`[ ]`, `( )`).
//!
//! Time: O(line length); no allocation per term.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{IndexError, Result};

/// An RDF term, borrowed from the input line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Term<'a> {
    /// IRI as written: `<http://...>` (with brackets) or a prefixed name `pfx:local`.
    /// The keyword `a` is returned as `rdf:type`.
    Iri(&'a str),
    /// Literal: raw lexical form (escapes kept), optional language tag, optional datatype.
    Literal {
        /// Raw lexical form between the quotes (or the bare token for numbers/booleans).
        lex: &'a str,
        /// Language tag without `@`.
        lang: Option<&'a str>,
        /// Datatype IRI as written (after `^^`).
        datatype: Option<&'a str>,
    },
}

/// One statement of a data line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Statement<'a> {
    /// `s p o .`
    Triple {
        /// Subject.
        s: Term<'a>,
        /// Predicate.
        p: Term<'a>,
        /// Object.
        o: Term<'a>,
    },
    /// `<< s p o >> mp mo .` (RDF-star annotation of a fact).
    Meta {
        /// Subject of the annotated fact.
        s: Term<'a>,
        /// Predicate of the annotated fact.
        p: Term<'a>,
        /// Object of the annotated fact.
        o: Term<'a>,
        /// Annotation predicate.
        mp: Term<'a>,
        /// Annotation value.
        mo: Term<'a>,
    },
}

/// Lexical token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tok<'a> {
    Iri(&'a str),
    Lit(Term<'a>),
    A,
    Semi,
    Comma,
    Dot,
    QOpen,
    QClose,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Directive,
}

/// Tokenizer over a string slice (one line or a whole schema file).
#[derive(Debug)]
pub(crate) struct Lexer<'a> {
    s: &'a str,
    b: &'a [u8],
    pos: usize,
    line: usize,
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n')
}

fn ends_name(c: u8) -> bool {
    is_ws(c)
        || matches!(
            c,
            b';' | b',' | b'"' | b'<' | b'(' | b')' | b'[' | b']' | b'#'
        )
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(s: &'a str, line: usize) -> Self {
        Self {
            s,
            b: s.as_bytes(),
            pos: 0,
            line,
        }
    }

    pub(crate) fn line(&self) -> usize {
        self.line
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(IndexError::Syntax {
            line: self.line,
            msg: msg.into(),
        })
    }

    fn peek(&self, k: usize) -> Option<u8> {
        self.b.get(self.pos + k).copied()
    }

    fn skip_ws_and_comments(&mut self) {
        while let Some(c) = self.peek(0) {
            if c == b'\n' {
                self.line += 1;
                self.pos += 1;
            } else if is_ws(c) {
                self.pos += 1;
            } else if c == b'#' {
                while let Some(c) = self.peek(0) {
                    if c == b'\n' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn iri_ref(&mut self) -> Result<&'a str> {
        let start = self.pos;
        self.pos += 1;
        while let Some(c) = self.peek(0) {
            if c == b'>' {
                self.pos += 1;
                return Ok(&self.s[start..self.pos]);
            }
            if is_ws(c) || c == b'<' {
                break;
            }
            self.pos += 1;
        }
        self.err("unterminated IRI")
    }

    fn name(&mut self) -> &'a str {
        let start = self.pos;
        while let Some(c) = self.peek(0) {
            if c == b'\\' {
                self.pos = (self.pos + 2).min(self.b.len());
                continue;
            }
            if ends_name(c) {
                break;
            }
            self.pos += 1;
        }
        // Local names cannot end with '.': it terminates the statement.
        while self.pos > start + 1 && self.b[self.pos - 1] == b'.' && self.b[self.pos - 2] != b'\\'
        {
            self.pos -= 1;
        }
        &self.s[start..self.pos]
    }

    fn literal(&mut self) -> Result<Term<'a>> {
        let q = self.b[self.pos];
        let lex;
        if self.peek(1) == Some(q) && self.peek(2) == Some(q) {
            let start = self.pos + 3;
            let mut i = start;
            loop {
                match self.b.get(i) {
                    None => return self.err("unterminated long literal"),
                    Some(b'\\') => i += 2,
                    Some(&c)
                        if c == q
                            && self.b.get(i + 1) == Some(&q)
                            && self.b.get(i + 2) == Some(&q) =>
                    {
                        break
                    }
                    Some(b'\n') => {
                        self.line += 1;
                        i += 1;
                    }
                    Some(_) => i += 1,
                }
            }
            lex = &self.s[start..i];
            self.pos = i + 3;
        } else {
            let start = self.pos + 1;
            let mut i = start;
            loop {
                match self.b.get(i) {
                    None | Some(b'\n') => return self.err("unterminated literal"),
                    Some(b'\\') => i += 2,
                    Some(&c) if c == q => break,
                    Some(_) => i += 1,
                }
            }
            if i > self.b.len() {
                return self.err("unterminated literal");
            }
            lex = &self.s[start..i];
            self.pos = i + 1;
        }
        let mut lang = None;
        let mut datatype = None;
        if self.peek(0) == Some(b'@') {
            self.pos += 1;
            let start = self.pos;
            while self
                .peek(0)
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'-')
            {
                self.pos += 1;
            }
            if self.pos == start {
                return self.err("empty language tag");
            }
            lang = Some(&self.s[start..self.pos]);
        } else if self.peek(0) == Some(b'^') && self.peek(1) == Some(b'^') {
            self.pos += 2;
            datatype = Some(match self.peek(0) {
                Some(b'<') => self.iri_ref()?,
                Some(c) if !ends_name(c) => self.name(),
                _ => return self.err("missing datatype"),
            });
        }
        Ok(Term::Literal {
            lex,
            lang,
            datatype,
        })
    }

    /// Next token, or `None` at the end of input.
    pub(crate) fn next_tok(&mut self) -> Result<Option<Tok<'a>>> {
        self.skip_ws_and_comments();
        let Some(c) = self.peek(0) else {
            return Ok(None);
        };
        let tok = match c {
            b'<' if self.peek(1) == Some(b'<') => {
                self.pos += 2;
                Tok::QOpen
            }
            b'<' => Tok::Iri(self.iri_ref()?),
            b'>' if self.peek(1) == Some(b'>') => {
                self.pos += 2;
                Tok::QClose
            }
            b'"' | b'\'' => Tok::Lit(self.literal()?),
            b';' | b',' | b'.' | b'[' | b']' | b'(' | b')' => {
                self.pos += 1;
                match c {
                    b';' => Tok::Semi,
                    b',' => Tok::Comma,
                    b'.' => Tok::Dot,
                    b'[' => Tok::LBracket,
                    b']' => Tok::RBracket,
                    b'(' => Tok::LParen,
                    _ => Tok::RParen,
                }
            }
            b'@' => {
                let word = self.name();
                if word == "@prefix" || word == "@base" {
                    Tok::Directive
                } else {
                    return self.err(format!("unknown directive `{word}`"));
                }
            }
            _ => {
                let word = self.name();
                if word.is_empty() {
                    return self.err(format!("unexpected character `{}`", c as char));
                }
                let first = word.as_bytes()[0];
                if word == "a" {
                    Tok::A
                } else if word == "true"
                    || word == "false"
                    || first.is_ascii_digit()
                    || ((first == b'+' || first == b'-')
                        && word.as_bytes().get(1).is_some_and(|d| d.is_ascii_digit()))
                {
                    Tok::Lit(Term::Literal {
                        lex: word,
                        lang: None,
                        datatype: None,
                    })
                } else if word.contains(':') {
                    Tok::Iri(word)
                } else {
                    return self.err(format!("unexpected token `{word}`"));
                }
            }
        };
        Ok(Some(tok))
    }
}

impl<'a> Lexer<'a> {
    /// Reads an IRI (or, if `allow_lit`, a literal) term.
    fn term(&mut self, what: &str, allow_lit: bool) -> Result<Term<'a>> {
        let tok = self.next_tok()?;
        self.as_term(tok, what, allow_lit)
    }

    fn as_term(&self, tok: Option<Tok<'a>>, what: &str, allow_lit: bool) -> Result<Term<'a>> {
        match tok {
            Some(Tok::Iri(s)) => Ok(Term::Iri(s)),
            Some(Tok::Lit(t)) if allow_lit => Ok(t),
            other => self.err(format!("expected {what}, found {other:?}")),
        }
    }

    /// Reads a predicate (`a` becomes `rdf:type`).
    fn verb(&mut self) -> Result<Term<'a>> {
        match self.next_tok()? {
            Some(Tok::Iri(s)) => Ok(Term::Iri(s)),
            Some(Tok::A) => Ok(Term::Iri("rdf:type")),
            other => self.err(format!("expected predicate, found {other:?}")),
        }
    }
}

fn parse_line_cb<'a>(
    line: &'a str,
    line_no: usize,
    emit: &mut dyn FnMut(Statement<'a>) -> Result<()>,
) -> Result<()> {
    let mut lx = Lexer::new(line, line_no);
    let first = match lx.next_tok()? {
        None | Some(Tok::Directive) => return Ok(()),
        Some(t) => t,
    };
    let (subject, quoted) = if first == Tok::QOpen {
        let s = lx.term("subject", false)?;
        let p = lx.verb()?;
        let o = lx.term("object", true)?;
        if lx.next_tok()? != Some(Tok::QClose) {
            return lx.err("expected `>>`");
        }
        (s, Some((s, p, o)))
    } else {
        (lx.as_term(Some(first), "subject", false)?, None)
    };
    loop {
        let p = lx.verb()?;
        let mut done = false;
        loop {
            let o = lx.term("object", true)?;
            emit(match quoted {
                Some((s, qp, qo)) => Statement::Meta {
                    s,
                    p: qp,
                    o: qo,
                    mp: p,
                    mo: o,
                },
                None => Statement::Triple { s: subject, p, o },
            })?;
            match lx.next_tok()? {
                Some(Tok::Comma) => continue,
                Some(Tok::Semi) => break,
                Some(Tok::Dot) => {
                    done = true;
                    break;
                }
                other => return lx.err(format!("expected `,`, `;` or `.`, found {other:?}")),
            }
        }
        if done {
            break;
        }
        // `;` may be followed directly by the final `.`.
        let save = (lx.pos, lx.line);
        if lx.next_tok()? == Some(Tok::Dot) {
            break;
        }
        (lx.pos, lx.line) = save;
    }
    match lx.next_tok()? {
        None => Ok(()),
        Some(t) => lx.err(format!("trailing token {t:?}")),
    }
}

/// Parses one data line; appends its statements to `out`.
/// Blank lines, comments and `@prefix` directives produce nothing.
pub fn parse_line<'a>(line: &'a str, line_no: usize, out: &mut Vec<Statement<'a>>) -> Result<()> {
    parse_line_cb(line, line_no, &mut |st| {
        out.push(st);
        Ok(())
    })
}

/// Calls `f` on every statement of a data file, line by line.
pub fn for_each_statement<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(&Statement<'_>) -> Result<()>,
{
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut buf = String::new();
    let mut line_no = 0usize;
    loop {
        buf.clear();
        line_no += 1;
        match reader.read_line(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                return Err(IndexError::Syntax {
                    line: line_no,
                    msg: "invalid UTF-8".into(),
                })
            }
            Err(e) => return Err(e.into()),
        }
        parse_line_cb(&buf, line_no, &mut |st| f(&st))?;
    }
}
