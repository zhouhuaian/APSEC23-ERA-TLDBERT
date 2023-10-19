import argparse
import logging
import time
import csv
from pathlib import Path
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_NAME = 'JiraRepos'

REPOS = ['Apache', 'Hyperledger', 'IntelDAOS', 'JFrog', 
        'Jira', 'JiraEcosystem', 'MariaDB', 'Mindville',
        'Mojang', 'MongoDB', 'Qt', 'RedHat', 
        'Sakai', 'SecondLife', 'Sonatype', 'Spring']

EPIC_LINKFIELD_DICT = {'Apache': 'customfield_12311120',
                        'Hyperledger': 'customfield_10006',
                        'IntelDAOS': 'customfield_10092',
                        'JFrog': 'customfield_10806',
                        'Jira': 'customfield_12931',
                        'JiraEcosystem': 'customfield_12180',
                        'MariaDB': 'customfield_10600',
                        'MindVille': 'customfield_10000',
                        'MongoDB': 'customfield_10857',
                        'Qt': 'customfield_10400',
                        'Redhat': 'customfield_12311140',
                        'Sakai': 'customfield_10772',
                        'SecondLife': 'customfield_10871',
                        'Sonatype': 'customfield_11500',
                        'Spring': 'customfield_10680'}

ISSUE_DIR = Path('../data/raw/issues')
ISSUE_DIR.mkdir(parents=True, exist_ok=True)
LINK_DIR = Path('../data/raw/links')
LINK_DIR.mkdir(parents=True, exist_ok=True)


def extract_issues_to_csv(repo: str, db):

    repo_collection = db[repo]
    cursor = repo_collection.find({})

    issues = []
    issue_ct = 0
    comment_ct = 0

    for issue in cursor:
        try:
            issue_key = issue['key']

            try:
                issuetype = issue['fields']['issuetype']['name']
            except Exception:
                issuetype = 'None'

            try:
                status = issue['fields']['status']['name']
            except Exception:
                status = 'None'
            
            try:
                priority = issue['fields']['priority']['name']
            except Exception:
                priority = 'None'
            
            projectid = issue['fields']['project']['name']

            try:
                resolution = issue['fields']['resolution']['name']
            except Exception:
                resolution = 'Open'

            try:
                componentsArray = issue['fields']['components']
                components = ' '
                for item in componentsArray:
                    components += item['name'] + ' '
            except Exception:
                components = ' '

            try:
                created = issue['fields']['created']
            except Exception:
                created = 'None'
            
            try:
                updated = issue['fields']['updated']
            except Exception:
                updated = 'None'

            try:
                title = issue['fields']['summary']
            except Exception:
                title = ' '

            try:
                description = issue['fields']['description']
            except Exception:
                description = ' '
            
            try:
                commentsArray = issue['fields']['comments']
                comments = ' '
                for item in commentsArray:
                    comments += item['body'] + ' '
                    comment_ct += 1
            except:
                comments = ' '

            issue_dict = {
                'issue_id': issue_key,
                'type': issuetype,
                'status': status,
                'priority': priority,
                'resolution': resolution,
                'project_id': projectid,
                'created': created,
                'updated': updated,
                'title': title,
                'description': description,
                'comments': comments,
                'components': components,
            }

            issues.append(issue_dict)
            issue_ct += 1

        except Exception as e:
            pass

    filename = ISSUE_DIR / (repo + '.csv')
    with open(filename, 'w', errors='surrogatepass', encoding='utf-8') as output_file:
        dict_wirter = csv.DictWriter(output_file, issues[0].keys(), delimiter=";")
        dict_wirter.writeheader()
        dict_wirter.writerows(issues)

    logging.info(f"Totally extracted {issue_ct} issues and {comment_ct} comments of {repo}")


def extract_links_to_csv(repo: str, db):

    repo_collection = db[repo]
    cursor = repo_collection.find({})

    links = []

    for issue in cursor:
        try:
            issue_key = issue['key']

            issuelinks = issue['fields']['issuelinks']

            for i in range(0, len(issuelinks)):
                issuelink = issuelinks[i]
                linktype = issuelink['type']['name']
                
                try:
                    in_issue = issue_key
                    out_issue = issuelink['outwardIssue']['key']
                except Exception:
                    in_issue = issuelink['inwardIssue']['key']
                    out_issue = issue_key

                name = in_issue + "_" + out_issue

                link_dict = {
                        'name': name,
                        'linktype': linktype,
                        'issue_id_1': in_issue,
                        'issue_id_2': out_issue,
                    }
                    
                links.append(link_dict)


            issuesubtasks = issue['fields']['subtasks']

            for i in range(0, len(issuesubtasks)):
                subtask = issuesubtasks[i]
                linktype = 'Subtask'
                in_issue = issue_key
                out_issue = subtask['key']

                name = in_issue + "_" + out_issue

                link_dict = {
                        'name': name,
                        'linktype': linktype,
                        'issue_id_1': in_issue,
                        'issue_id_2': out_issue,
                    }
                    
                links.append(link_dict)


            try:
                epic = issue['fields'][EPIC_LINKFIELD_DICT[repo]]
                in_issue = issue_key
                out_issue = epic
                name = in_issue + "_" + out_issue
                linktype = 'Epic-Relation'

                link_dict = {
                        'name': name,
                        'linktype': linktype,
                        'issue_id_1': in_issue,
                        'issue_id_2': out_issue,
                    }
                    
                links.append(link_dict)
            
            except Exception:
                pass

            
            if repo == 'RedHat':
                try:
                    parent = issue['fields']['customfield_12313140']
                    in_issue = issue_key
                    out_issue = parent
                    name = in_issue + "_" + out_issue
                    linktype = 'Parent-Relation'

                    link_dict = {
                        'name': name,
                        'linktype': linktype,
                        'issue_id_1': in_issue,
                        'issue_id_2': out_issue,
                    }
                    
                    links.append(link_dict)

                except Exception:
                    pass

                try:
                    feature = issue['fields']['customfield_12318341']
                    in_issue = issue_key
                    out_issue = feature
                    name = in_issue + "_" + out_issue
                    linktype = 'Feature-Relation'
                    
                    link_dict = {
                        'name': name,
                        'linktype': linktype,
                        'issue_id_1': in_issue,
                        'issue_id_2': out_issue,
                    }
                    
                    links.append(link_dict)
                
                except Exception:
                    pass
        
        except Exception as e:
            pass
    
    filename = LINK_DIR / (repo + '.csv')
    with open(filename, 'w', errors='surrogatepass', encoding='utf-8') as output_file:
            dict_wirter = csv.DictWriter(output_file, links[0].keys(), delimiter=";")
            dict_wirter.writeheader()
            dict_wirter.writerows(links)
    
    logging.info(f"Extracting links for {repo} done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract issues data')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=27017)
    parser.add_argument('--username', default=None)
    parser.add_argument('--password', default=None)
    args = parser.parse_args()

    start_time = time.perf_counter()

    with MongoClient(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        serverSelectionTimeoutMS=5000,
    ) as client:
        db = client[DB_NAME]
        
        for repo in REPOS:
            extract_issues_to_csv(repo, db)
            extract_links_to_csv(repo, db)

    end_time = time.perf_counter()

    logging.info(f"Time cost: {end_time - start_time} s")