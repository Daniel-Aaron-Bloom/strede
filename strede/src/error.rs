/// Error types that can be constructed by derived [`Deserialize`](crate::Deserialize) implementations.
///
/// Derived implementations add `where D::Error: DeserializeError` to their
/// [`Deserialize::deserialize`](crate::Deserialize::deserialize) methods so they can report duplicate map keys.
pub trait DeserializeError: Sized {
    /// A map key appeared more than once.
    fn duplicate_field(field: &'static str) -> Self;
}
