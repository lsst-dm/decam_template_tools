# Config override for selectTemplateCoaddVisits, loaded via -C in
# bps_build_template.yaml. This lives in a file rather than a -c option because
# bps treats {...} in YAML values as variable placeholders and silently
# substitutes an empty string, which turns a dict literal into "name=".
config.visitSummaryMinValues = {"effTimeZeroPointScale": 0.75}
