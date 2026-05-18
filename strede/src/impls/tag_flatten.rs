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

pub struct TagAwareMap<'de, 'v, M, const N: usize>
where
    M: MapAccess<'de>,
{
    inner: M,
    tag_field: &'static str,
    tag_candidates: [(&'static str, usize); N],
    expected_variant: usize,
    tag_value: &'v core::cell::Cell<Option<usize>>,
    _phantom: PhantomData<&'de ()>,
}

impl<'de, 'v, M, const N: usize> TagAwareMap<'de, 'v, M, N>
where
    M: MapAccess<'de>,
{
    pub fn new(
        inner: M,
        tag_field: &'static str,
        tag_candidates: [(&'static str, usize); N],
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

impl<'de, 'v, M, const N: usize> MapAccess<'de> for TagAwareMap<'de, 'v, M, N>
where
    M: MapAccess<'de>,
    Match: Deserialize<
            'de,
            <M::KeyProbe as MapKeyProbe<'de>>::KeySubDeserializer,
            Extra = &'static str,
        >,
    MatchVals<usize, N>: Deserialize<
            'de,
            <crate::borrow::VP<'de, M::KeyProbe> as MapValueProbe<'de>>::ValueSubDeserializer,
            Extra = [(&'static str, usize); N],
        >,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        Self {
            inner: self.inner.fork(),
            tag_field: self.tag_field,
            tag_candidates: self.tag_candidates,
            expected_variant: self.expected_variant,
            tag_value: self.tag_value,
            _phantom: PhantomData,
        }
    }

    async fn iterate<S>(self, arms: S) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStack<'de, M::KeyProbe>,
    {
        let tf = self.tag_field;
        let tc = self.tag_candidates;
        let expected = self.expected_variant;
        let tv = self.tag_value;
        let wrapped = crate::TagInjectingStackOwned::new(
            arms,
            tf,
            tc,
            tv,
            move |kp: M::KeyProbe| kp.deserialize_key::<Match>(tf),
            move |vp: crate::borrow::VP<'de, M::KeyProbe>| {
                vp.deserialize_value::<MatchVals<usize, N>>(tc)
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
// FlattenMapAccess — MapAccess adapter that splices outer arms into the
// inner type's `iterate` call.
//
// The outer struct's `deserialize_from_map` body constructs its own arms,
// hands a `FlattenMapAccess(inner_map, outer_arms, outer_outputs_cell, cont)`
// to the flatten field's `DeserializeFromMap::deserialize_from_map`, and
// after that returns reads `outer_outputs_cell` to finalize its own fields.
//
// `Cont` is the [`crate::FlattenCont`] continuation that runs after the
// outer and inner arm stacks are concatenated. For a single flatten field
// it is [`crate::FlattenTerminal`] (or [`crate::FlattenTerminalBoxed`]); for
// multi-flatten the derive macro emits one intermediate continuation struct
// per flatten level, chaining into the next field's
// `DeserializeFromMap` impl until the terminal.
// -------------------------------------------------------------------------

pub struct FlattenMapAccess<'de, 'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccess<'de>,
{
    inner: M,
    outer_arms: Option<OuterArms>,
    outer_outputs: &'c core::cell::Cell<Option<OuterOut>>,
    cont: Option<Cont>,
    _phantom: PhantomData<&'de ()>,
}

impl<'de, 'c, M, OuterArms, OuterOut, Cont> FlattenMapAccess<'de, 'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccess<'de>,
{
    pub fn new(
        inner: M,
        outer_arms: OuterArms,
        outer_outputs: &'c core::cell::Cell<Option<OuterOut>>,
        cont: Cont,
    ) -> Self {
        Self {
            inner,
            outer_arms: Some(outer_arms),
            outer_outputs,
            cont: Some(cont),
            _phantom: PhantomData,
        }
    }
}

impl<'de, 'c, M, OuterArms, OuterOut, Cont> MapAccess<'de>
    for FlattenMapAccess<'de, 'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccess<'de>,
    OuterArms: crate::MapArmStack<'de, M::KeyProbe, Outputs = OuterOut>,
    Cont: crate::FlattenCont<'de, M>,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        unreachable!("FlattenMapAccess cannot be forked")
    }

    async fn iterate<S>(
        mut self,
        inner_arms: S,
    ) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStack<'de, M::KeyProbe>,
    {
        let outer = self
            .outer_arms
            .take()
            .expect("FlattenMapAccess::iterate called twice");
        let cont = self
            .cont
            .take()
            .expect("FlattenMapAccess::iterate called twice");
        let combined = crate::StackConcat(outer, inner_arms);
        match cont.finish(self.inner, combined).await? {
            Probe::Hit((claim, (outer_outs, inner_outs))) => {
                self.outer_outputs.set(Some(outer_outs));
                Ok(Probe::Hit((claim, inner_outs)))
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

// -------------------------------------------------------------------------
// TagAwareMapOwned — owned counterpart to TagAwareMap.
// -------------------------------------------------------------------------

pub struct TagAwareMapOwned<'v, M, const N: usize>
where
    M: MapAccessOwned,
{
    inner: M,
    tag_field: &'static str,
    tag_candidates: [(&'static str, usize); N],
    expected_variant: usize,
    tag_value: &'v core::cell::Cell<Option<usize>>,
}

impl<'v, M, const N: usize> TagAwareMapOwned<'v, M, N>
where
    M: MapAccessOwned,
{
    pub fn new(
        inner: M,
        tag_field: &'static str,
        tag_candidates: [(&'static str, usize); N],
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

impl<'v, M, const N: usize> MapAccessOwned for TagAwareMapOwned<'v, M, N>
where
    M: MapAccessOwned,
    Match: DeserializeOwned<
            <M::KeyProbe as MapKeyProbeOwned>::KeySubDeserializer,
            Extra = &'static str,
        >,
    MatchVals<usize, N>: DeserializeOwned<
            <crate::owned::VP<M::KeyProbe> as MapValueProbeOwned>::ValueSubDeserializer,
            Extra = [(&'static str, usize); N],
        >,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        Self {
            inner: self.inner.fork(),
            tag_field: self.tag_field,
            tag_candidates: self.tag_candidates,
            expected_variant: self.expected_variant,
            tag_value: self.tag_value,
        }
    }

    async fn iterate<S>(self, arms: S) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStackOwned<M::KeyProbe>,
    {
        let tf = self.tag_field;
        let tc = self.tag_candidates;
        let expected = self.expected_variant;
        let tv = self.tag_value;
        let wrapped = crate::TagInjectingStackOwned::new(
            arms,
            tf,
            tc,
            tv,
            move |kp: M::KeyProbe| kp.deserialize_key::<Match>(tf),
            move |vp: crate::owned::VP<M::KeyProbe>| {
                vp.deserialize_value::<MatchVals<usize, N>>(tc)
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
// FlattenMapAccessOwned — owned counterpart to FlattenMapAccess.
// -------------------------------------------------------------------------

pub struct FlattenMapAccessOwned<'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccessOwned,
{
    inner: M,
    outer_arms: Option<OuterArms>,
    outer_outputs: &'c core::cell::Cell<Option<OuterOut>>,
    cont: Option<Cont>,
}

impl<'c, M, OuterArms, OuterOut, Cont> FlattenMapAccessOwned<'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccessOwned,
{
    pub fn new(
        inner: M,
        outer_arms: OuterArms,
        outer_outputs: &'c core::cell::Cell<Option<OuterOut>>,
        cont: Cont,
    ) -> Self {
        Self {
            inner,
            outer_arms: Some(outer_arms),
            outer_outputs,
            cont: Some(cont),
        }
    }
}

impl<'c, M, OuterArms, OuterOut, Cont> MapAccessOwned
    for FlattenMapAccessOwned<'c, M, OuterArms, OuterOut, Cont>
where
    M: MapAccessOwned,
    OuterArms: crate::MapArmStackOwned<M::KeyProbe, Outputs = OuterOut>,
    Cont: crate::FlattenContOwned<M>,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        unreachable!("FlattenMapAccessOwned cannot be forked")
    }

    async fn iterate<S>(
        mut self,
        inner_arms: S,
    ) -> Result<Probe<(M::MapClaim, S::Outputs)>, M::Error>
    where
        S: crate::MapArmStackOwned<M::KeyProbe>,
    {
        let outer = self
            .outer_arms
            .take()
            .expect("FlattenMapAccessOwned::iterate called twice");
        let cont = self
            .cont
            .take()
            .expect("FlattenMapAccessOwned::iterate called twice");
        let combined = crate::StackConcat(outer, inner_arms);
        match cont.finish(self.inner, combined).await? {
            Probe::Hit((claim, (outer_outs, inner_outs))) => {
                self.outer_outputs.set(Some(outer_outs));
                Ok(Probe::Hit((claim, inner_outs)))
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}
