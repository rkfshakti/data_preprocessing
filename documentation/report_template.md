# Report Template

What the template does

- `src/report_template.html` is a Jinja2 template that assembles the pieces produced by the pipeline into a single-page, tabbed report.
- It embeds Plotly fragments (optionally including the Plotly runtime for offline viewing), static images, and small HTML snippets for tables (top-10 rows) while linking to full CSVs for download.

Sections

- Executive summary (KPIs and top positive/negative terms)
- Trends (sentiment over time, keyword trends)
- Multivariate (predictive terms, topic sentiment correlation, co-occurrence)
- Clusters (cluster terms and samples)

Customization

- You can edit the template to reorganize sections, include additional charts, or add download links.
- If you want to change the number of rows shown inline, modify the `.head(10)` usage in `run_analysis.py`.

Notes

- The template uses a minimal tab implementation with small JS placed at the end of the document to ensure event handlers attach after DOM creation.
- Plotly fragments are generated in `run_analysis.py` via `plotly.io.to_html(..., full_html=False)` and inserted into the template.
