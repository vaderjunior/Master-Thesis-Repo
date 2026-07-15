from src.hsrag.data.record import Record

r = Record(
    id="hatexplain-12345",
    text="Go back to your own country, nobody wants you here.",
    lang="en",
    source="hatexplain",
    gate=True,
    target_groups=["national_origin"],
    hate_types=None,      # HateXplain doesn't annotate hate_type
    severity=None,        # nor severity
    raw={"annotator_labels": ["hatespeech", "hatespeech", "offensive"]},
)

print(r)