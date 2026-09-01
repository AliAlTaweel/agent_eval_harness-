import json
import sys

import click

from agent.graph import ReviewFailedError, run_review
from agent.llm import OllamaClient
from agent.render import render_comment


@click.group()
def cli():
    pass


@cli.command()
@click.option("--diff-file", required=True, type=click.Path(exists=True), help="Path to a diff/patch file.")
@click.option("--trace-out", default="trace.json", type=click.Path(), help="Where to write the RunTrace JSON.")
@click.option("--comment-out", default="comment.md", type=click.Path(), help="Where to write the markdown PR comment.")
@click.option("--model", default="qwen2.5:7b", help="Ollama model to use.")
def review(diff_file, trace_out, comment_out, model):
    with open(diff_file) as f:
        diff_text = f.read()
    client = OllamaClient(model=model)

    try:
        verdict, trace = run_review(diff=diff_text, client=client)
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
