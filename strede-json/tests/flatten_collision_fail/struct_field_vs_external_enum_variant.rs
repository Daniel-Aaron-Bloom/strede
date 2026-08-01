//! The exact motivating case from TESTING_GAPS.md: a struct field's wire name
//! collides with one of a flattened externally-tagged enum's variant names.
use strede_derive::Deserialize;

#[derive(Deserialize)]
struct Pagination {
    limit: u64,
    offset: u64,
    total: u64,
}

#[derive(Deserialize)]
enum Message {
    Request { id: String, method: String },
    Users {
        users: Vec<String>,
        id: u32,
        #[strede(flatten)]
        pagination: Pagination,
    },
}

#[derive(Deserialize)]
struct Users {
    #[allow(non_snake_case)]
    Users: Vec<String>, // collides with the `Users` variant name of Message
    #[strede(flatten)]
    message: Message,
}

fn main() {}
