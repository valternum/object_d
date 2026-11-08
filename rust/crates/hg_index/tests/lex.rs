//! Issue #6: YAGO 4.6 Turtle reader. Fixture lines are copied from the
//! yago-4.6 samples (tiny / meta), TAB-separated as in the files.

use hg_index::lex::{for_each_statement, parse_line, Statement, Term};
use hg_index::IndexError;
use proptest::prelude::*;

fn iri(s: &str) -> Term<'_> {
    Term::Iri(s)
}

fn parse(line: &str) -> Vec<Statement<'_>> {
    let mut out = Vec::new();
    parse_line(line, 1, &mut out).unwrap();
    out
}

fn triple<'a>(s: Term<'a>, p: Term<'a>, o: Term<'a>) -> Statement<'a> {
    Statement::Triple { s, p, o }
}

#[test]
fn typed_literal() {
    let got = parse("yago:Belgium\tyago:populationNumber\t\"+11584008\"^^xsd:decimal\t.");
    let o = Term::Literal {
        lex: "+11584008",
        lang: None,
        datatype: Some("xsd:decimal"),
    };
    assert_eq!(
        got,
        vec![triple(iri("yago:Belgium"), iri("yago:populationNumber"), o)]
    );
}

#[test]
fn language_tagged_literal() {
    let got = parse("yago:Belgium\trdfs:label\t\"Bélgié\"@jv\t.");
    let o = Term::Literal {
        lex: "Bélgié",
        lang: Some("jv"),
        datatype: None,
    };
    assert_eq!(got, vec![triple(iri("yago:Belgium"), iri("rdfs:label"), o)]);
}

#[test]
fn escaped_backslash_before_closing_quote() {
    let line = "yago:Bachelor_of_Medicine_Bachelor_of_Surgery\trdfs:label\t\"бакалавр медицины/хирургии\\\\\"@ru\t.";
    let got = parse(line);
    let o = Term::Literal {
        lex: "бакалавр медицины/хирургии\\\\",
        lang: Some("ru"),
        datatype: None,
    };
    assert_eq!(
        got,
        vec![triple(
            iri("yago:Bachelor_of_Medicine_Bachelor_of_Surgery"),
            iri("rdfs:label"),
            o
        )]
    );
}

#[test]
fn escaped_quotes_inside_literal() {
    let got = parse("yago:A\tschema:slogan\t\"say \\\"hi\\\"\"@en\t.");
    let o = Term::Literal {
        lex: "say \\\"hi\\\"",
        lang: Some("en"),
        datatype: None,
    };
    assert_eq!(got, vec![triple(iri("yago:A"), iri("schema:slogan"), o)]);
}

#[test]
fn long_literal() {
    let got = parse("yago:A rdfs:comment \"\"\"two \"quoted\" words\"\"\"@en .");
    let o = Term::Literal {
        lex: "two \"quoted\" words",
        lang: Some("en"),
        datatype: None,
    };
    assert_eq!(got, vec![triple(iri("yago:A"), iri("rdfs:comment"), o)]);
}

#[test]
fn abbreviated_line_with_semicolon() {
    let got = parse("yago:Bahraini_dinar rdfs:label \"yago:Bahraini_dinar\"@en; rdf:type yago:UnitOfMeasurement .");
    let label = Term::Literal {
        lex: "yago:Bahraini_dinar",
        lang: Some("en"),
        datatype: None,
    };
    assert_eq!(
        got,
        vec![
            triple(iri("yago:Bahraini_dinar"), iri("rdfs:label"), label),
            triple(
                iri("yago:Bahraini_dinar"),
                iri("rdf:type"),
                iri("yago:UnitOfMeasurement")
            ),
        ]
    );
}

#[test]
fn object_list_with_comma_and_keyword_a() {
    let got = parse("yago:A a schema:Person; schema:memberOf yago:B, yago:C .");
    assert_eq!(
        got,
        vec![
            triple(iri("yago:A"), iri("rdf:type"), iri("schema:Person")),
            triple(iri("yago:A"), iri("schema:memberOf"), iri("yago:B")),
            triple(iri("yago:A"), iri("schema:memberOf"), iri("yago:C")),
        ]
    );
}

#[test]
fn meta_fact() {
    let line = "<<\tyago:Belgium\tschema:location\tyago:European_Coal_and_Steel_Community\t>>\tschema:startDate\t\"1951\"^^xsd:gYear\t.";
    let got = parse(line);
    assert_eq!(
        got,
        vec![Statement::Meta {
            s: iri("yago:Belgium"),
            p: iri("schema:location"),
            o: iri("yago:European_Coal_and_Steel_Community"),
            mp: iri("schema:startDate"),
            mo: Term::Literal {
                lex: "1951",
                lang: None,
                datatype: Some("xsd:gYear")
            },
        }]
    );
}

#[test]
fn full_iris_and_local_names_with_dots() {
    let got = parse("<http://x/a> <http://x/p> yago:St._Louis .");
    assert_eq!(
        got,
        vec![triple(
            iri("<http://x/a>"),
            iri("<http://x/p>"),
            iri("yago:St._Louis")
        )]
    );
}

#[test]
fn bare_number_literal() {
    let got = parse("ys:p sh:maxCount 1 .");
    let o = Term::Literal {
        lex: "1",
        lang: None,
        datatype: None,
    };
    assert_eq!(got, vec![triple(iri("ys:p"), iri("sh:maxCount"), o)]);
}

#[test]
fn directives_comments_and_blank_lines_yield_nothing() {
    assert!(parse("@prefix yago: <http://yago-knowledge.org/resource/> .").is_empty());
    assert!(parse("# a comment").is_empty());
    assert!(parse("   \t  ").is_empty());
    assert!(parse("").is_empty());
    assert_eq!(parse("yago:A yago:p yago:B . # trailing").len(), 1);
}

#[test]
fn malformed_lines_are_errors_with_line_numbers() {
    for bad in [
        "yago:A yago:p \"unterminated",
        "yago:A yago:p .",
        "yago:A yago:p yago:B",
        "yago:A yago:p yago:B yago:C .",
        "<< yago:A yago:p yago:B yago:q yago:C .",
        "yago:A <http://unterminated yago:B .",
    ] {
        let mut out = Vec::new();
        match parse_line(bad, 7, &mut out) {
            Err(IndexError::Syntax { line, .. }) => assert_eq!(line, 7, "{bad}"),
            other => panic!("expected a syntax error for `{bad}`, got {other:?}"),
        }
    }
}

#[test]
fn file_reader_reports_statements_and_first_error_line() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("holm_lex_{}.ttl", std::process::id()));
    std::fs::write(
        &path,
        "@prefix yago: <http://yago-knowledge.org/resource/> .\r\nyago:A\tyago:p\tyago:B\t.\r\nyago:A yago:q \"x\"@en; yago:r yago:C .\r\n",
    )
    .unwrap();
    let mut n = 0;
    for_each_statement(&path, |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 3);

    std::fs::write(&path, "yago:A yago:p yago:B .\nyago:A yago:p .\n").unwrap();
    let err = for_each_statement(&path, |_| Ok(())).unwrap_err();
    assert!(matches!(err, IndexError::Syntax { line: 2, .. }), "{err:?}");
    std::fs::remove_file(&path).unwrap();
}

proptest! {
    /// Arbitrary input never panics.
    #[test]
    fn never_panics(s in "\\PC{0,80}") {
        let mut out = Vec::new();
        let _ = parse_line(&s, 1, &mut out);
    }

    /// Arbitrary Turtle-ish token soup never panics.
    #[test]
    fn token_soup_never_panics(parts in proptest::collection::vec(
        prop_oneof![
            Just("yago:A"), Just("<http://x>"), Just("\"lit\""), Just("@en"), Just("^^xsd:int"),
            Just(";"), Just(","), Just("."), Just("<<"), Just(">>"), Just("a"), Just("\\"), Just("\""),
        ], 0..12)) {
        let line = parts.join(" ");
        let mut out = Vec::new();
        let _ = parse_line(&line, 1, &mut out);
    }
}
