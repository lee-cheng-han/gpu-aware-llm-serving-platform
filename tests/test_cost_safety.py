from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_runtime_dependencies_do_not_include_paid_service_sdks():
    requirements = "\n".join(
        path.read_text().lower() for path in ROOT.glob("requirements*.txt")
    )
    prohibited = {
        "anthropic",
        "boto3",
        "google-cloud-aiplatform",
        "groq",
        "openai",
        "replicate",
        "stripe",
        "together",
    }
    assert prohibited.isdisjoint(requirements.split())


def test_hosted_ci_is_manual_only_and_cannot_deploy_or_publish():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text().lower()
    assert "workflow_dispatch:" in workflow
    assert "\n  push:" not in workflow
    assert "pull_request:" not in workflow
    assert "push: false" in workflow
    assert "kubectl" not in workflow
    assert "nvidia.com/gpu" not in workflow


def test_default_kubernetes_documentation_uses_context_guard():
    documentation = (ROOT / "k8s/README.md").read_text()
    guard = (ROOT / "scripts/deploy_local_kind.sh").read_text()
    assert "make deploy-kind" in documentation
    assert 'expected_context="kind-llm-inference"' in guard
