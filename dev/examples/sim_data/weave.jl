using Weave

# weave("simulate_data.jmd"; doctype = "md2pdf", out_path = :pwd)

weave("simulate_data.jmd"; doctype = "github", out_path = :pwd)