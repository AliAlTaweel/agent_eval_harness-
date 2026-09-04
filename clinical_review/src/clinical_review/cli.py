import json
import sys

import click

from clinical_review.graph import ReviewFailedError, run_review
from clinical_review.llm import OllamaClient
from clinical_review.render import render_comment


@click.group()
def cli():
    pass


@cli.command()
@click.option("--note-file", required=True, type=click.Path(exists=True), help="Path to a clinical note text file.")
@click.option(
    "--encounter-type",
    required=True,
    type=click.Choice(["new_patient", "follow_up"]),
    help="Encounter type — determines which documentation elements are required.",
)
@click.option("--trace-out", default="trace.json", type=click.Path(), help="Where to write the RunTrace JSON.")
@click.option(
    "--comment-out", default="comment.md", type=click.Path(), help="Where to write the markdown review comment."
)
@click.option("--model", default="qwen2.5:7b", help="Ollama model to use.")
def review(note_file, encounter_type, trace_out, comment_out, model):
    with open(note_file) as f:
        note_text = f.read()
    client = OllamaClient(model=model)

    try:
        verdict, trace = run_review(note_text=note_text, encounter_type=encounter_type, client=client)
    except ReviewFailedError as exc:
        with open(trace_out, "w") as f:
            json.dump(exc.trace.model_dump(mode="json"), f, indent=2)
        click.echo(f"Review failed: {exc}", err=True)
        sys.exit(1)

    with open(trace_out, "w") as f:
        json.dump(trace.model_dump(mode="json"), f, indent=2)

    with open(comment_out, "w") as f:
        f.write(render_comment(verdict))

    click.echo(json.dumps(verdict.model_dump(), indent=2))


if __name__ == "__main__":
    cli()
