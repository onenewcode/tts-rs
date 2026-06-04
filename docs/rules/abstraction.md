# Abstraction Rules

Only extract a method, helper, trait, wrapper, or module when the abstraction
removes real complexity or protects a real boundary.

Do not introduce an abstraction only to make simple code look organized. Keep
straight-line code inline when it is short, local, and easier to read at the
call site.

An abstraction is justified when at least one of these is true:

- The same non-trivial logic is repeated in multiple places.
- A named boundary clarifies ownership between crates, modules, or layers.
- The extracted unit can be tested independently and has meaningful behavior.
- The caller is hiding details that would otherwise obscure the main flow.
- The abstraction reduces future change risk without widening the public API.

Avoid extracting when:

- The body is only one or two obvious statements.
- The helper name merely repeats the code it contains.
- There is only one caller and the extraction does not clarify the surrounding
  logic.
- The abstraction would force extra cloning, allocation, lifetimes, generics, or
  trait indirection for no concrete benefit.

When unsure, keep the code inline first. Extract later only after the need is
visible in the code.
