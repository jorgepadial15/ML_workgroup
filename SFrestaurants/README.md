This project is for playing with [San-Francisco Restaurant Scores data](https://data.sfgov.org/Health-and-Social-Services/Restaurant-Scores-LIVES-Standard/pyih-qa8i?row_index=0)

### Before start working on the project, please do
```
git checkout SFrestaurants                   # switch the branch from master to SF restaurants     
git pull --rebase							 # copy changes from the remote rep and append them your local rep (no merge commit)
git pull --no-commit						 # if the previous command causes issues, try this one
```
For details see [this tutorial](https://www.atlassian.com/git/tutorials/syncing/git-pull)

## Tentative goal
  - Predict the inspection score based on the location of the restaurant
  - Predict the dates and routes for inspection (?) to get the most effective results (to observe most of the voilations)

## Setting environment
- To improve reproducibility of the project I've created environment.yml file so now everyone can create the same conda environment (details in the resources.md)

## How to commit
1. Stage changes for commit
```
git add <file>                  			 # stages changes in <file> for commit     
```
or 
```
git add -A                 					 # stages all files in the directory for commit (new, modified, deleted)     
```
2. Commit to the local rep

```
git commit -m "fixed file1"					 # commits staged changes with commit message on your local machine 
```

3. Upload to remote rep
```
git push						 			# uploads commits to a remote repository
```
