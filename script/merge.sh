#!/bin/bash

# 브랜치 목록
branches=("plotify-ver1" "plotify-ver3" "plotify-ver4" "plotify-ver5" "plotify-ver6" "plotify-ver7" "plotify-ver8" "plotify-ver9" "plotify-ver10")

# 메인 브랜치에서 변경사항 가져오기
git checkout main
git pull origin main

# 각 브랜치로 변경사항 병합
for branch in "${branches[@]}"
do
    echo "Merging main into $branch"
    git checkout "$branch"
    git merge main
    git add .
    git commit -m "Merged main into $branch"
    git push origin "$branch"
done

