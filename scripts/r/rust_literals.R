# Helpers that print R values as Rust literals, shared by the R reference generators.

# A float as a Rust literal that round-trips exactly.
rust_float <- function(x) {
  if (is.na(x)) return("f64::NAN")
  # The shortest of 15, 16 or 17 significant digits that reads back as the same double.
  for (digits in 15:17) {
    s <- sprintf(paste0("%.", digits, "g"), x)
    if (as.numeric(s) == x) break
  }
  if (!grepl("[.eE]", s)) s <- paste0(s, ".0")
  if (grepl("^-?inf", s, ignore.case = TRUE)) return(if (x > 0) "f64::INFINITY" else "f64::NEG_INFINITY")
  s
}

# Floats as a Rust slice literal.
rust_array <- function(values) {
  values <- as.numeric(values)
  if (length(values) == 0) return("&[]")
  paste0("&[", paste(vapply(values, rust_float, ""), collapse = ", "), "]")
}

# Integers as a Rust slice literal.
rust_usize_array <- function(values) {
  if (length(values) == 0) return("&[]")
  paste0("&[", paste(as.integer(values), collapse = ", "), "]")
}
