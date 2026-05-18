//! Borrow-family `ValueBorrowed` and `RawValueBorrowed` fixtures.

#![cfg(all(feature = "alloc", not(feature = "arbitrary_precision")))]

use strede::Probe;
use strede_json::{JsonDeserializer, RawValueBorrowed, ValueBorrowed};
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de str) -> T
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    let probe = block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
    match probe {
        Probe::Hit((_, v)) => v,
        Probe::Miss => panic!("Miss"),
    }
}

#[test]
fn value_null() {
    let v: ValueBorrowed<'_> = parse("null");
    assert_eq!(v, ValueBorrowed::Null);
}

#[test]
fn value_bool_true() {
    let v: ValueBorrowed<'_> = parse("true");
    assert_eq!(v, ValueBorrowed::Bool(true));
}

#[test]
fn value_bool_false() {
    let v: ValueBorrowed<'_> = parse("false");
    assert_eq!(v, ValueBorrowed::Bool(false));
}

#[test]
fn value_number_int() {
    let v: ValueBorrowed<'_> = parse("42");
    match v {
        ValueBorrowed::Number(n) => assert_eq!(n.as_u64(), Some(42)),
        _ => panic!("expected Number"),
    }
}

#[test]
fn value_string() {
    let v: ValueBorrowed<'_> = parse(r#""hello""#);
    match v {
        ValueBorrowed::String(s) => assert_eq!(s, "hello"),
        _ => panic!("expected String"),
    }
}

#[test]
fn value_array_empty() {
    let v: ValueBorrowed<'_> = parse("[]");
    assert_eq!(v, ValueBorrowed::Array(vec![]));
}

#[test]
fn value_array_mixed() {
    let v: ValueBorrowed<'_> = parse(r#"[1, "two", true, null]"#);
    match v {
        ValueBorrowed::Array(items) => {
            assert_eq!(items.len(), 4);
            assert!(matches!(items[0], ValueBorrowed::Number(_)));
            assert!(matches!(items[1], ValueBorrowed::String(ref s) if s == "two"));
            assert_eq!(items[2], ValueBorrowed::Bool(true));
            assert_eq!(items[3], ValueBorrowed::Null);
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn value_object_empty() {
    let v: ValueBorrowed<'_> = parse("{}");
    assert_eq!(v, ValueBorrowed::Object(vec![]));
}

#[test]
fn value_object_flat() {
    let v: ValueBorrowed<'_> = parse(r#"{"a": 1, "b": "two"}"#);
    match v {
        ValueBorrowed::Object(pairs) => {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].0, "a");
            assert!(matches!(pairs[0].1, ValueBorrowed::Number(_)));
            assert_eq!(pairs[1].0, "b");
            assert!(matches!(pairs[1].1, ValueBorrowed::String(ref s) if s == "two"));
        }
        _ => panic!("expected Object"),
    }
}

#[test]
fn value_nested() {
    let v: ValueBorrowed<'_> = parse(r#"{"outer": {"inner": [1, 2, 3]}, "flag": false}"#);
    match v {
        ValueBorrowed::Object(pairs) => {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].0, "outer");
            match &pairs[0].1 {
                ValueBorrowed::Object(inner) => {
                    assert_eq!(inner[0].0, "inner");
                    match &inner[0].1 {
                        ValueBorrowed::Array(a) => assert_eq!(a.len(), 3),
                        _ => panic!("expected inner array"),
                    }
                }
                _ => panic!("expected inner object"),
            }
            assert_eq!(pairs[1].1, ValueBorrowed::Bool(false));
        }
        _ => panic!("expected Object"),
    }
}

#[test]
fn raw_value_in_struct() {
    use strede_derive::Deserialize;

    #[derive(Deserialize)]
    struct Wrap<'de> {
        id: u32,
        #[strede(borrow)]
        raw: RawValueBorrowed<'de>,
    }

    let s = r#"{"id": 1, "raw": {"nested": [1, 2, 3]}}"#;
    let de = JsonDeserializer::new(s.as_bytes());
    let probe = block_on(<Wrap<'_> as strede::Deserialize<'_, _>>::deserialize(
        de,
        (),
    ))
    .unwrap();
    match probe {
        Probe::Hit((_, w)) => {
            assert_eq!(w.id, 1);
            assert_eq!(w.raw.as_str(), r#"{"nested": [1, 2, 3]}"#);
        }
        Probe::Miss => panic!("Miss"),
    }
}
