from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer
from pr_agent.dicl.sdk import DICL
from pr_agent.algo.pr_processing import get_pr_diff
from pr_agent.algo.token_handler import TokenHandler


class DICLEvolve:
    def __init__(self, pr_url: str, ai_handler=None, args: list = None):
        self.pr_url = pr_url
        self.args = args or []
        self.ai_handler = ai_handler

    async def run(self):
        get_settings().set("enable_dicl", True)
        get_settings().set("config.publish_output", False)

        reviewer = PRReviewer(self.pr_url, args=self.args, ai_handler=self.ai_handler)
        git_provider = reviewer.git_provider

        model = get_settings().config.model
        token_handler = TokenHandler(git_provider.pr, reviewer.vars, model)
        diff_files = get_pr_diff(git_provider, token_handler, model)

        changed_files = []
        if isinstance(diff_files, list):
            changed_files = [
                file.filename if hasattr(file, "filename") else file
                for file in diff_files
                if isinstance(file, str) or hasattr(file, "filename")
            ]
        elif isinstance(diff_files, str):
            changed_files = [diff_files]

        pr_data = {
            "title": git_provider.pr.title,
            "description": reviewer.pr_description,
            "changed_files": changed_files,
            "language": reviewer.main_language,
            "pr_id": f"{git_provider.repo}/{git_provider.get_pr_id()}",
            "diff": diff_files,
        }

        description = (
            pr_data["description"][:500] + "..."
            if len(pr_data["description"]) > 500
            else pr_data["description"]
        )

        base_prompt = f"""## PR Review Request
        Title: {pr_data['title']}
        Language: {pr_data['language']}
        Files: {', '.join(pr_data['changed_files'])}

        Description: {description}

        Review this PR for issues, improvements, and best practices."""

        enhanced_review = await DICL.evolve(pr_data, base_prompt)

        print("\n" + "=" * 80)
        print("DICL Enhanced PR Review with Auto-Learning")
        print("=" * 80)

        return enhanced_review
