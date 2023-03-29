# Email-2-FAQ

## Contributing Guidelines

### Setting Up
Fork the repository and clone it on your system.
  ```bash
  git clone <repo-url>
  ```

### Pull Requests
1. For each new `submission`, `fix` or `feature` create a new branch named `<github-handle>-<explanatory-name>`.
    ```bash
    git branch <branch-name>
    ```
1. Switch to the new branch.
    ```bash
    git checkout <branch-name>
    ```
1. Make the changes in the new branch.
1. Stage the changes for the next commit.
   To stage changes from specific files:
    ```bash
    git add <filename>
    ```
    To stage all the changes at once:
    ```bash
    git add .
    ```
    Use `git status` to track the changes made.
1. Commit the changes.
    ```bash
    git commit -m "<commit-message>"
    ```
1. Push the changes to your forked repo. If you're working on a new branch:
    ```bash
    git push -u origin <branch-name>
    ```
    If the branch already exists:
    ```bash
    git push
    ```
1. Create a `pull request`.

### Keep in Mind

- Use `meaningful small commits`. Refer to this [link](https://cbea.ms/git-commit/).
- `Remember to fetch` changes from the upstream repo before working on something.