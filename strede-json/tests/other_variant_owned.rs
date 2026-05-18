//! Owned-family `#[strede(other)]` fixtures.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Color {
    Red,
    Green,
    #[strede(other)]
    Other,
}

fn parse(input: &[u8]) -> Color {
    block_on_loop(SharedBuf::with_async(
        input,
        async |buf: &mut &[u8]| {
            *buf = &[];
        },
        async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            match <Color as DeserializeOwned<_>>::deserialize_owned(de, ())
                .await
                .unwrap()
            {
                Probe::Hit((_, v)) => v,
                Probe::Miss => panic!("Miss"),
            }
        },
    ))
}

#[test]
fn known_variant() {
    assert_eq!(parse(b"\"Red\""), Color::Red);
    assert_eq!(parse(b"\"Green\""), Color::Green);
}

#[test]
fn unknown_falls_to_other() {
    assert_eq!(parse(b"\"Blue\""), Color::Other);
    assert_eq!(parse(b"\"Yellow\""), Color::Other);
}
