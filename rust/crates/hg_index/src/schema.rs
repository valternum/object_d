//! Parser of the YAGO SHACL schema (`yago-schema.ttl`).
//!
//! A small Turtle document parser (multi-line statements, `;` / `,`,
//! anonymous blank nodes `[ ... ]`, collections `( ... )`) produces triples;
//! they are then interpreted as class shapes and property constraints:
//! domain = classes whose shape lists the property, range = `sh:class` /
//! `sh:datatype` (also inside `sh:or` lists), `sh:maxCount`.

use std::collections::{BTreeMap, HashMap};

use crate::error::{IndexError, Result};
use crate::lex::{Lexer, Term, Tok};

/// A node shape (class) of the schema.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassShape {
    /// Direct superclasses (`rdfs:subClassOf`), sorted.
    pub supers: Vec<String>,
    /// Disjoint classes (`owl:disjointWith`), sorted.
    pub disjoint: Vec<String>,
}

/// Constraints of one property (`sh:path`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PropertyInfo {
    /// Classes whose shape lists the property, sorted.
    pub domain: Vec<String>,
    /// Range classes (`sh:class`, also inside `sh:or`), sorted.
    pub range_classes: Vec<String>,
    /// Range datatypes (`sh:datatype`, also inside `sh:or`), sorted.
    pub range_datatypes: Vec<String>,
    /// `sh:maxCount`, if any.
    pub max_count: Option<u32>,
}

/// The parsed schema.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schema {
    /// Class shapes by class name.
    pub classes: BTreeMap<String, ClassShape>,
    /// Properties by path.
    pub properties: BTreeMap<String, PropertyInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    Iri(String),
    Lit(String),
    Blank(u32),
}

struct Parser<'a> {
    lx: Lexer<'a>,
    peeked: Option<Option<Tok<'a>>>,
    next_blank: u32,
    triples: Vec<(Node, String, Node)>,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(IndexError::Syntax {
            line: self.lx.line(),
            msg: msg.into(),
        })
    }

    fn next(&mut self) -> Result<Option<Tok<'a>>> {
        match self.peeked.take() {
            Some(t) => Ok(t),
            None => self.lx.next_tok(),
        }
    }

    fn peek(&mut self) -> Result<Option<Tok<'a>>> {
        if self.peeked.is_none() {
            self.peeked = Some(self.lx.next_tok()?);
        }
        Ok(self.peeked.expect("just filled"))
    }

    fn blank(&mut self) -> Node {
        self.next_blank += 1;
        Node::Blank(self.next_blank)
    }

    fn document(&mut self) -> Result<()> {
        loop {
            match self.next()? {
                None => return Ok(()),
                Some(Tok::Directive) => loop {
                    match self.next()? {
                        Some(Tok::Dot) => break,
                        None => return self.err("unterminated directive"),
                        Some(_) => {}
                    }
                },
                Some(Tok::Iri(s)) => self.predicate_objects(&Node::Iri(s.to_owned()), Tok::Dot)?,
                Some(Tok::LBracket) => {
                    let b = self.blank_props()?;
                    if self.peek()? == Some(Tok::Dot) {
                        self.next()?;
                    } else {
                        self.predicate_objects(&b, Tok::Dot)?;
                    }
                }
                Some(t) => return self.err(format!("unexpected {t:?} at statement start")),
            }
        }
    }

    /// `verb objects (';' verb objects)* closer`
    fn predicate_objects(&mut self, subject: &Node, closer: Tok<'a>) -> Result<()> {
        loop {
            let verb = match self.next()? {
                Some(Tok::Iri(s)) => s.to_owned(),
                Some(Tok::A) => "rdf:type".to_owned(),
                other => return self.err(format!("expected predicate, found {other:?}")),
            };
            loop {
                let o = self.object()?;
                self.triples.push((subject.clone(), verb.clone(), o));
                if self.peek()? == Some(Tok::Comma) {
                    self.next()?;
                } else {
                    break;
                }
            }
            match self.next()? {
                Some(Tok::Semi) => {
                    if self.peek()? == Some(closer) {
                        self.next()?;
                        return Ok(());
                    }
                }
                Some(t) if t == closer => return Ok(()),
                other => return self.err(format!("expected `;` or {closer:?}, found {other:?}")),
            }
        }
    }

    fn blank_props(&mut self) -> Result<Node> {
        let b = self.blank();
        if self.peek()? == Some(Tok::RBracket) {
            self.next()?;
        } else {
            self.predicate_objects(&b, Tok::RBracket)?;
        }
        Ok(b)
    }

    fn object(&mut self) -> Result<Node> {
        match self.next()? {
            Some(Tok::Iri(s)) => Ok(Node::Iri(s.to_owned())),
            Some(Tok::Lit(Term::Literal { lex, .. })) => Ok(Node::Lit(lex.to_owned())),
            Some(Tok::LBracket) => self.blank_props(),
            Some(Tok::LParen) => {
                let mut items = Vec::new();
                while self.peek()? != Some(Tok::RParen) {
                    if self.peek()?.is_none() {
                        return self.err("unterminated collection");
                    }
                    items.push(self.object()?);
                }
                self.next()?;
                let mut tail = Node::Iri("rdf:nil".to_owned());
                for item in items.into_iter().rev() {
                    let cell = self.blank();
                    self.triples
                        .push((cell.clone(), "rdf:first".to_owned(), item));
                    self.triples
                        .push((cell.clone(), "rdf:rest".to_owned(), tail));
                    tail = cell;
                }
                Ok(tail)
            }
            other => self.err(format!("expected object, found {other:?}")),
        }
    }
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Parses the text of a YAGO schema file.
pub fn parse_schema(text: &str) -> Result<Schema> {
    let mut p = Parser {
        lx: Lexer::new(text, 1),
        peeked: None,
        next_blank: 0,
        triples: Vec::new(),
    };
    p.document()?;

    let mut by_subject: HashMap<Node, Vec<(String, Node)>> = HashMap::new();
    for (s, pred, o) in p.triples {
        by_subject.entry(s).or_default().push((pred, o));
    }
    let objects = |s: &Node, pred: &str| -> Vec<Node> {
        by_subject
            .get(s)
            .map(|v| {
                v.iter()
                    .filter(|(q, _)| q == pred)
                    .map(|(_, o)| o.clone())
                    .collect()
            })
            .unwrap_or_default()
    };
    let iris = |nodes: Vec<Node>| -> Vec<String> {
        nodes
            .into_iter()
            .filter_map(|n| match n {
                Node::Iri(s) => Some(s),
                _ => None,
            })
            .collect()
    };

    let mut schema = Schema::default();
    let mut shape_path: HashMap<String, String> = HashMap::new();
    let mut subjects: Vec<&Node> = by_subject.keys().collect();
    subjects.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    for s in &subjects {
        let Node::Iri(name) = s else { continue };
        if let Some(Node::Iri(path)) = objects(s, "sh:path").into_iter().next() {
            shape_path.insert(name.clone(), path.clone());
            let info = schema.properties.entry(path).or_default();
            info.range_classes.extend(iris(objects(s, "sh:class")));
            info.range_datatypes.extend(iris(objects(s, "sh:datatype")));
            for list in objects(s, "sh:or") {
                let mut cell = list;
                while let Node::Blank(_) = cell {
                    for item in objects(&cell, "rdf:first") {
                        info.range_classes.extend(iris(objects(&item, "sh:class")));
                        info.range_datatypes
                            .extend(iris(objects(&item, "sh:datatype")));
                    }
                    cell = objects(&cell, "rdf:rest")
                        .into_iter()
                        .next()
                        .unwrap_or(Node::Iri("rdf:nil".into()));
                }
            }
            for m in objects(s, "sh:maxCount") {
                let Node::Lit(lex) = m else {
                    return Err(IndexError::Schema(format!(
                        "{name}: sh:maxCount must be a literal"
                    )));
                };
                let n = lex
                    .parse::<u32>()
                    .map_err(|_| IndexError::Schema(format!("{name}: bad sh:maxCount `{lex}`")))?;
                info.max_count = Some(info.max_count.map_or(n, |m| m.min(n)));
            }
        }
    }
    for s in &subjects {
        let Node::Iri(name) = s else { continue };
        if !objects(s, "rdf:type").contains(&Node::Iri("sh:NodeShape".into())) {
            continue;
        }
        schema.classes.insert(
            name.clone(),
            ClassShape {
                supers: sorted(iris(objects(s, "rdfs:subClassOf"))),
                disjoint: sorted(iris(objects(s, "owl:disjointWith"))),
            },
        );
        for shape in iris(objects(s, "sh:property")) {
            if let Some(path) = shape_path.get(&shape) {
                schema
                    .properties
                    .entry(path.clone())
                    .or_default()
                    .domain
                    .push(name.clone());
            }
        }
    }
    for info in schema.properties.values_mut() {
        info.domain = sorted(std::mem::take(&mut info.domain));
        info.range_classes = sorted(std::mem::take(&mut info.range_classes));
        info.range_datatypes = sorted(std::mem::take(&mut info.range_datatypes));
    }
    Ok(schema)
}
