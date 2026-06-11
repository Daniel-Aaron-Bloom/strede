use super::*;
use crate::borrow::{MapAccess, MapKeyProbe, MapValueProbe};
use crate::owned::{MapAccessOwned, MapKeyProbeOwned, MapValueProbeOwned};

// -------------------------------------------------------------------------
// TagAwareMap — MapAccess adapter that injects a tag arm and post-validates
// the matched variant index. The caller hands the *map* to the variant
// payload's `DeserializeFromMap::deserialize_from_map` impl, which runs its
// own `iterate` — TagAwareMap wraps that `iterate` to prepend the tag arm
// and check the captured variant index against `expected_variant`.
// -------------------------------------------------------------------------

pub struct TagAwareMap<'de, 'v, M, W>
where
    M: MapAccess<'de>,
{
    inner: M,
    tag_field: &'static str,
    tag_candidates: W,
    expected_variant: usize,
    tag_value: &'v core::cell::Cell<Option<usize>>,
    _phantom: PhantomData<&'de ()>,
}

impl<'de, 'v, M, W> TagAwareMap<'de, 'v, M, W>
where
    M: MapAccess<'de>,
{
    pub fn new(
        inner: M,
        tag_field: &'static str,
        tag_candidates: W,
        expected_variant: usize,
        tag_value: &'v core::cell::Cell<Option<usize>>,
    ) -> Self {
        Self {
            inner,
            tag_field,
            tag_candidates,
            expected_variant,
            tag_value,
            _phantom: PhantomData,
        }
    }
}

impl<'de, 'v, M, W> MapAccess<'de> for TagAwareMap<'de, 'v, M, W>
where
    M: MapAccess<'de>,
    W: Copy,
    Match: Deserialize<
            'de,
            <M::KeyProbe as MapKeyProbe<'de>>::KeySubDeserializer,
            Extra = &'static str,
        >,
    MatchVals<usize, W>: Deserialize<
            'de,
            <crate::borrow::VP<'de, M::KeyProbe> as MapValueProbe<'de>>::ValueSubDeserializer,
            Extra = W,
        >,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    async fn iterate<S>(self, arms: S) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStack<'de, M::KeyProbe>,
    {
        let tf = self.tag_field;
        let tc = self.tag_candidates;
        let expected = self.expected_variant;
        let tv = self.tag_value;
        let wrapped = crate::TagInjectingStack::new(
            arms,
            tf,
            tc,
            tv,
            move |kp: M::KeyProbe, _i: usize| kp.deserialize_key::<Match>(tf),
            move |vp: crate::borrow::VP<'de, M::KeyProbe>| {
                vp.deserialize_value::<MatchVals<usize, W>>(tc)
            },
        );
        match self.inner.iterate(wrapped).await? {
            Probe::Hit((claim, outs)) => {
                if tv.get() == Some(expected) {
                    Ok(Probe::Hit((claim, outs)))
                } else {
                    Ok(Probe::Miss)
                }
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}


// -------------------------------------------------------------------------
// TagAwareMapOwned — owned counterpart to TagAwareMap.
// -------------------------------------------------------------------------

pub struct TagAwareMapOwned<'v, M, W>
where
    M: MapAccessOwned,
{
    inner: M,
    tag_field: &'static str,
    tag_candidates: W,
    expected_variant: usize,
    tag_value: &'v core::cell::Cell<Option<usize>>,
}

impl<'v, M, W> TagAwareMapOwned<'v, M, W>
where
    M: MapAccessOwned,
{
    pub fn new(
        inner: M,
        tag_field: &'static str,
        tag_candidates: W,
        expected_variant: usize,
        tag_value: &'v core::cell::Cell<Option<usize>>,
    ) -> Self {
        Self {
            inner,
            tag_field,
            tag_candidates,
            expected_variant,
            tag_value,
        }
    }
}

impl<'v, M, W> MapAccessOwned for TagAwareMapOwned<'v, M, W>
where
    M: MapAccessOwned,
    W: Copy,
    Match: DeserializeOwned<
            <M::KeyProbe as MapKeyProbeOwned>::KeySubDeserializer,
            Extra = &'static str,
        >,
    MatchVals<usize, W>: DeserializeOwned<
            <crate::owned::VP<M::KeyProbe> as MapValueProbeOwned>::ValueSubDeserializer,
            Extra = W,
        >,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    async fn iterate<S>(self, arms: S) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStackOwned<M::KeyProbe>,
    {
        let tf = self.tag_field;
        let tc = self.tag_candidates;
        let expected = self.expected_variant;
        let tv = self.tag_value;
        let wrapped = crate::TagInjectingStack::new(
            arms,
            tf,
            tc,
            tv,
            move |kp: M::KeyProbe, _i: usize| kp.deserialize_key::<Match>(tf),
            move |vp: crate::owned::VP<M::KeyProbe>| {
                vp.deserialize_value::<MatchVals<usize, W>>(tc)
            },
        );
        match self.inner.iterate(wrapped).await? {
            Probe::Hit((claim, outs)) => {
                if tv.get() == Some(expected) {
                    Ok(Probe::Hit((claim, outs)))
                } else {
                    Ok(Probe::Miss)
                }
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

