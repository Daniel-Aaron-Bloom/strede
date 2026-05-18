//! JSON value types: [`ValueBorrowed`], [`ValueOwned`], [`RawValueBorrowed`], [`RawValueOwned`].
//!
//! Each value type ships a single-concrete-D `Deserialize` / `DeserializeOwned`
//! impl bound to the JSON sub-deserializer. By the orphan rule and the v3
//! design, this restricts any containing struct that uses one of these types
//! to the JSON family — the trait solver enforces `SubDeserializer =
//! JsonSubDeserializer<'de>` (or the chunked equivalent) at the use site.

extern crate alloc;

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

#[allow(unused_imports)]
use strede::{
    Buffer, Deserialize, DeserializeOwned, Probe,
    borrow::{
        Deserializer as _, Entry as _, MapAccess as _, MapKeyClaim, MapKeyProbe, MapValueProbe,
    },
    hit,
    owned::{
        DeserializerOwned as _, EntryOwned as _, MapAccessOwned as _, MapKeyClaimOwned,
        MapKeyProbeOwned, MapValueProbeOwned,
    },
    select_probe,
};

use crate::JsonError;
use crate::chunked::{ChunkedJsonClaim, ChunkedJsonSubDeserializer, capture_raw_value_chunked};
use crate::full::{JsonClaim, JsonSubDeserializer};
use crate::number::{NumberBorrowed, NumberOwned};

// ===========================================================================
// ValueBorrowed
// ===========================================================================

#[derive(Debug, PartialEq)]
pub enum ValueBorrowed<'de> {
    Null,
    Bool(bool),
    Number(NumberBorrowed<'de>),
    String(Cow<'de, str>),
    Array(Vec<ValueBorrowed<'de>>),
    Object(Vec<(Cow<'de, str>, ValueBorrowed<'de>)>),
}

#[cfg(not(feature = "arbitrary_precision"))]
impl<'de> Deserialize<'de, JsonSubDeserializer<'de>> for ValueBorrowed<'de> {
    type Extra = ();
    async fn deserialize(
        d: JsonSubDeserializer<'de>,
        _: (),
    ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
        d.entry(|[e1, e2, e3, e4, e5, e6]| async move {
            select_probe! {
                async move {
                    let (c, _) = hit!(e1.deserialize_value::<()>(()).await);
                    Ok(Probe::Hit((c, ValueBorrowed::Null)))
                },
                async move {
                    let (c, b) = hit!(e2.deserialize_value::<bool>(()).await);
                    Ok(Probe::Hit((c, ValueBorrowed::Bool(b))))
                },
                async move {
                    let (c, n) = hit!(e3.deserialize_value::<NumberBorrowed<'de>>(()).await);
                    Ok(Probe::Hit((c, ValueBorrowed::Number(n))))
                },
                async move {
                    let (c, s) = hit!(e4.deserialize_value::<Cow<'de, str>>(()).await);
                    Ok(Probe::Hit((c, ValueBorrowed::String(s))))
                },
                async move {
                    let r: Result<Probe<(JsonClaim<'de>, Vec<ValueBorrowed<'de>>)>, JsonError> =
                        alloc::boxed::Box::pin(e5
                            .deserialize_seq_into::<Vec<ValueBorrowed<'de>>>(()))
                        .await;
                    let (c, v) = hit!(r);
                    Ok(Probe::Hit((c, ValueBorrowed::Array(v))))
                },
                async move {
                    let map = hit!(e6.deserialize_map().await);
                    let r: Result<
                        Probe<(JsonClaim<'de>, Vec<(Cow<'de, str>, ValueBorrowed<'de>)>)>,
                        JsonError,
                    > = alloc::boxed::Box::pin(map.iterate(CollectObject::new())).await;
                    let (c, out) = hit!(r);
                    Ok(Probe::Hit((c, ValueBorrowed::Object(out))))
                },
            }
        })
        .await
    }
}

// Top-level `Deserialize<'de, JsonDeserializer<'de>>` impl so callers can write
// `<ValueBorrowed as Deserialize<'_, _>>::deserialize(JsonDeserializer::new(..), ())`.
// JsonDeserializer's `Entry::SubDeserializer = JsonSubDeserializer<'de>`, so this
// forwards through the entry to the impl above.
#[cfg(not(feature = "arbitrary_precision"))]
impl<'de> Deserialize<'de, crate::JsonDeserializer<'de>> for ValueBorrowed<'de> {
    type Extra = ();
    async fn deserialize(
        d: crate::JsonDeserializer<'de>,
        _: (),
    ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
        d.entry(|[e]| async move { e.deserialize_value::<ValueBorrowed<'de>>(()).await })
            .await
    }
}

/// Arm stack that collects all key-value pairs into a `Vec<(Cow<'de, str>, ValueBorrowed<'de>)>`.
#[cfg(not(feature = "arbitrary_precision"))]
struct CollectObject<'de> {
    pending_key: Option<Cow<'de, str>>,
    out: Vec<(Cow<'de, str>, ValueBorrowed<'de>)>,
}

#[cfg(not(feature = "arbitrary_precision"))]
impl<'de> CollectObject<'de> {
    fn new() -> Self {
        Self {
            pending_key: None,
            out: Vec::new(),
        }
    }
}

#[cfg(not(feature = "arbitrary_precision"))]
impl<'de, KP> strede::MapArmStack<'de, KP> for CollectObject<'de>
where
    KP: MapKeyProbe<'de>,
    Cow<'de, str>: Deserialize<'de, KP::KeySubDeserializer, Extra = ()>,
    ValueBorrowed<'de>: Deserialize<
            'de,
            <<KP::KeyClaim as MapKeyClaim<'de>>::ValueProbe as MapValueProbe<'de>>::ValueSubDeserializer,
            Extra = (),
        >,
{
    const SIZE: usize = 1;
    type Outputs = Vec<(Cow<'de, str>, ValueBorrowed<'de>)>;

    fn unsatisfied_count(&self) -> usize {
        0
    }
    fn open_count(&self) -> usize {
        1
    }

    type RaceState = ();
    fn init_race(&mut self, _: KP) {
        unreachable!()
    }
    fn poll_race_one(
        &mut self,
        _: core::pin::Pin<&mut ()>,
        _: usize,
        _: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!()
    }

    type DispatchState = ();
    fn init_dispatch(
        &mut self,
        _: usize,
        _: <KP::KeyClaim as MapKeyClaim<'de>>::ValueProbe,
    ) {
        unreachable!()
    }
    fn poll_dispatch(
        &mut self,
        _: core::pin::Pin<&mut ()>,
        _: &mut core::task::Context<'_>,
    ) -> core::task::Poll<
        Result<
            Probe<(
                <<KP::KeyClaim as MapKeyClaim<'de>>::ValueProbe as MapValueProbe<'de>>::ValueClaim,
                (),
            )>,
            KP::Error,
        >,
    > {
        unreachable!()
    }

    async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
        let (kc, k) = hit!(kp.deserialize_key::<Cow<'de, str>>(()).await);
        self.pending_key = Some(k);
        Ok(Probe::Hit((0, kc)))
    }

    async fn dispatch_value(
        &mut self,
        _: usize,
        vp: <KP::KeyClaim as MapKeyClaim<'de>>::ValueProbe,
    ) -> Result<
        Probe<(
            <<KP::KeyClaim as MapKeyClaim<'de>>::ValueProbe as MapValueProbe<'de>>::ValueClaim,
            (),
        )>,
        KP::Error,
    > {
        let k = self
            .pending_key
            .take()
            .expect("dispatch without pending key");
        let (vc, v) = hit!(vp.deserialize_value::<ValueBorrowed<'de>>(()).await);
        self.out.push((k, v));
        Ok(Probe::Hit((vc, ())))
    }

    fn take_outputs(&mut self) -> Self::Outputs {
        core::mem::take(&mut self.out)
    }
}

// ===========================================================================
// ValueOwned
// ===========================================================================

#[derive(Debug, PartialEq)]
pub enum ValueOwned {
    Null,
    Bool(bool),
    Number(NumberOwned),
    String(String),
    Array(Vec<ValueOwned>),
    Object(Vec<(String, ValueOwned)>),
}

impl<'s, B, F> DeserializeOwned<ChunkedJsonSubDeserializer<'s, B, F>> for ValueOwned
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
{
    type Extra = ();
    async fn deserialize_owned(
        d: ChunkedJsonSubDeserializer<'s, B, F>,
        _: (),
    ) -> Result<Probe<(ChunkedJsonClaim<'s, B, F>, Self)>, JsonError> {
        d.entry(|[e1, e2, e3, e4, e5, e6]| async move {
            select_probe! {
                async move {
                    let (c, _) = hit!(e1.deserialize_value::<()>(()).await);
                    Ok(Probe::Hit((c, ValueOwned::Null)))
                },
                async move {
                    let (c, b) = hit!(e2.deserialize_value::<bool>(()).await);
                    Ok(Probe::Hit((c, ValueOwned::Bool(b))))
                },
                async move {
                    let (c, n) = hit!(e3.deserialize_value::<NumberOwned>(()).await);
                    Ok(Probe::Hit((c, ValueOwned::Number(n))))
                },
                async move {
                    let (c, s) = hit!(e4.deserialize_value::<String>(()).await);
                    Ok(Probe::Hit((c, ValueOwned::String(s))))
                },
                async move {
                    let r: Result<Probe<(ChunkedJsonClaim<'s, B, F>, Vec<ValueOwned>)>, JsonError> =
                        alloc::boxed::Box::pin(e5.deserialize_seq_into::<Vec<ValueOwned>>(())).await;
                    let (c, v) = hit!(r);
                    Ok(Probe::Hit((c, ValueOwned::Array(v))))
                },
                async move {
                    let map = hit!(e6.deserialize_map().await);
                    let r: Result<
                        Probe<(ChunkedJsonClaim<'s, B, F>, Vec<(String, ValueOwned)>)>,
                        JsonError,
                    > = alloc::boxed::Box::pin(map.iterate(CollectObjectOwned::new())).await;
                    let (c, out) = hit!(r);
                    Ok(Probe::Hit((c, ValueOwned::Object(out))))
                },
            }
        })
        .await
    }
}

// Top-level entry point through ChunkedJsonDeserializer.
impl<'s, B, F> DeserializeOwned<crate::chunked::ChunkedJsonDeserializer<'s, B, F>> for ValueOwned
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
{
    type Extra = ();
    async fn deserialize_owned(
        d: crate::chunked::ChunkedJsonDeserializer<'s, B, F>,
        _: (),
    ) -> Result<Probe<((), Self)>, JsonError> {
        d.entry(|[e]| async move { e.deserialize_value::<ValueOwned>(()).await })
            .await
    }
}

struct CollectObjectOwned {
    pending_key: Option<String>,
    out: Vec<(String, ValueOwned)>,
}

impl CollectObjectOwned {
    fn new() -> Self {
        Self {
            pending_key: None,
            out: Vec::new(),
        }
    }
}

impl<KP> strede::MapArmStackOwned<KP> for CollectObjectOwned
where
    KP: MapKeyProbeOwned,
    String: DeserializeOwned<KP::KeySubDeserializer, Extra = ()>,
    ValueOwned: DeserializeOwned<
            <<KP::KeyClaim as MapKeyClaimOwned>::ValueProbe as MapValueProbeOwned>::ValueSubDeserializer,
            Extra = (),
        >,
{
    const SIZE: usize = 1;
    type Outputs = Vec<(String, ValueOwned)>;

    fn unsatisfied_count(&self) -> usize {
        0
    }
    fn open_count(&self) -> usize {
        1
    }

    type RaceState = ();
    fn init_race(&mut self, _: KP) {
        unreachable!()
    }
    fn poll_race_one(
        &mut self,
        _: core::pin::Pin<&mut ()>,
        _: usize,
        _: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!()
    }

    type DispatchState = ();
    fn init_dispatch(
        &mut self,
        _: usize,
        _: <KP::KeyClaim as MapKeyClaimOwned>::ValueProbe,
    ) {
        unreachable!()
    }
    fn poll_dispatch(
        &mut self,
        _: core::pin::Pin<&mut ()>,
        _: &mut core::task::Context<'_>,
    ) -> core::task::Poll<
        Result<
            Probe<(
                <<KP::KeyClaim as MapKeyClaimOwned>::ValueProbe as MapValueProbeOwned>::ValueClaim,
                (),
            )>,
            KP::Error,
        >,
    > {
        unreachable!()
    }

    async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
        let (kc, k) = hit!(kp.deserialize_key::<String>(()).await);
        self.pending_key = Some(k);
        Ok(Probe::Hit((0, kc)))
    }

    async fn dispatch_value(
        &mut self,
        _: usize,
        vp: <KP::KeyClaim as MapKeyClaimOwned>::ValueProbe,
    ) -> Result<
        Probe<(
            <<KP::KeyClaim as MapKeyClaimOwned>::ValueProbe as MapValueProbeOwned>::ValueClaim,
            (),
        )>,
        KP::Error,
    > {
        let k = self
            .pending_key
            .take()
            .expect("dispatch without pending key");
        let (vc, v) = hit!(vp.deserialize_value::<ValueOwned>(()).await);
        self.out.push((k, v));
        Ok(Probe::Hit((vc, ())))
    }

    fn take_outputs(&mut self) -> Self::Outputs {
        core::mem::take(&mut self.out)
    }
}

// ===========================================================================
// RawValueBorrowed
// ===========================================================================

/// Raw JSON bytes captured without parsing, borrowed from the source buffer.
///
/// JSON-format-only. Implementing `Deserialize<'de, JsonSubDeserializer<'de>>`
/// (one concrete `D`) ties any containing struct to the JSON sub-deserializer
/// — attempting to deserialize through a different format's sub-deserializer
/// fails to compile.
///
/// ```
/// # use strede::Deserialize;
/// # use strede_json::{JsonSubDeserializer, RawValueBorrowed};
/// fn _ok<'de>()
/// where RawValueBorrowed<'de>: Deserialize<'de, JsonSubDeserializer<'de>> {}
/// ```
///
/// ```compile_fail
/// # use strede::{Deserialize, JsonError};
/// # use strede_json::RawValueBorrowed;
/// // `Never` is a generic Deserializer that's not the JSON sub-deserializer.
/// // The bound below is unsatisfiable.
/// fn _bad<'de>()
/// where RawValueBorrowed<'de>: Deserialize<'de, strede::Never<'de, (), strede_json::JsonError>> {}
/// ```
#[repr(transparent)]
#[derive(Debug, PartialEq)]
pub struct RawValueBorrowed<'de>(pub &'de [u8]);

impl<'de> RawValueBorrowed<'de> {
    pub fn as_bytes(&self) -> &'de [u8] {
        self.0
    }
    /// JSON source is always valid UTF-8.
    pub fn as_str(&self) -> &'de str {
        unsafe { core::str::from_utf8_unchecked(self.0) }
    }
}

impl<'de> Deserialize<'de, JsonSubDeserializer<'de>> for RawValueBorrowed<'de> {
    type Extra = ();
    async fn deserialize(
        d: JsonSubDeserializer<'de>,
        _: (),
    ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
        let (start_src, src, token) = d.into_raw_source();
        let mut cur = src;
        let claim_tok = crate::full::skip_value(&mut cur, token)?;
        // `start_src` may include whitespace that the tokenizer skipped before
        // the leading value byte; strip it so the raw slice begins at the
        // first non-whitespace byte of the value.
        let mut head = start_src;
        while let Some((&b, rest)) = head.split_first() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                head = rest;
            } else {
                break;
            }
        }
        let consumed = head.len() - cur.len();
        let raw = &head[..consumed];
        Ok(Probe::Hit((
            JsonClaim {
                tokenizer: claim_tok,
                src: cur,
            },
            RawValueBorrowed(raw),
        )))
    }
}

// ===========================================================================
// RawValueOwned
// ===========================================================================

/// Raw JSON bytes captured from a streaming source, as an owned allocation.
#[repr(transparent)]
#[derive(Debug, PartialEq)]
pub struct RawValueOwned(pub Vec<u8>);

impl RawValueOwned {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn as_str(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.0) }
    }
}

impl<'s, B, F> DeserializeOwned<ChunkedJsonSubDeserializer<'s, B, F>> for RawValueOwned
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
{
    type Extra = ();
    async fn deserialize_owned(
        d: ChunkedJsonSubDeserializer<'s, B, F>,
        _: (),
    ) -> Result<Probe<(ChunkedJsonClaim<'s, B, F>, Self)>, JsonError> {
        let (handle, mut tokenizer, mut offset, start_offset, pending_tok) = d.into_raw_source();
        // `into_raw_source` always returns `Some(tok)` because the sub-de wraps
        // a pre-loaded token; the chunked entry path never empties it.
        let tok = pending_tok.expect("ChunkedJsonSubDeserializer with no pending token");
        let mut out = Vec::new();
        let handle = capture_raw_value_chunked(
            handle,
            &mut tokenizer,
            &mut offset,
            start_offset,
            tok,
            &mut out,
        )
        .await?;
        // Strip leading whitespace that the tokenizer skipped before the value.
        let trim = out
            .iter()
            .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            .count();
        if trim > 0 {
            out.drain(..trim);
        }
        Ok(Probe::Hit((
            ChunkedJsonClaim {
                tokenizer,
                offset,
                handle,
            },
            RawValueOwned(out),
        )))
    }
}
