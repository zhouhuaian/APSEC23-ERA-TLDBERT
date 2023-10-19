# REPOS=(Apache Hyperledger IntelDAOS JFrog Jira \
#         JiraEcosystem MariaDB Mojang MongoDB Qt \
#         RedHat Sakai SecondLife Sonatype Spring)

repo=Apache

nohup python finetune.py \
        --model bert-base-uncased \
        --tracker $repo \
        --train-batch-size 48 \
        --eval-batch-size 128 \
        --n-epochs 30 > ../../tmp/log/bert_run/${repo}_run.log 2>&1 &

# nohup python finetune.py \
#         --model ../../tmp/tldbert \
#         --tracker $repo \
#         --train-batch-size 48 \
#         --eval-batch-size 128 \
#         --n-epochs 30 > ../../tmp/log/tldbert_run/${repo}_run.log 2>&1 &