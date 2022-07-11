
# Create issues
## Create EPIC
jira.create_issue(fields=dict(summary=epic,
                             project = dict(key=myproj['key']),
                            description=f'Epic:{epic}',
                             issuetype = dict(name='Epic'),
                             ))
## Create issue               
jira.create_issue(fields=dict(summary=i['Name'].replace('\n',''),
                             project = dict(key=i['JiraProject']),
                            description=i['Description'].replace('<br>','')+'\n Roadmap Ticket: {}'.format(i['Item ID']),
                             issuetype = dict(name=i['Type']),
                              labels=mylabels,
                              parent=dict(key=EPIC_key),
                            assignee=dict(id=myassignee)
                             ))
## Create subtask
 jira.create_issue(fields=dict(summary=ticket_title,
                             project = dict(key='DATA'),
                             issuetype = dict(name='Subtask'),
                              parent=dict(key=parent_ticket)
                             ))

# Change / reassign parent
newParentKey='DATA-14'
jira.update_issue_field('DATA-447', fields={'parent':{'key':newParentKey}})


# Delete issues
jira.delete_issue('DATA-1')


# Add a link (URL)
jira.create_or_update_issue_remote_links(issue_key, "https://sedex-my.sharepoint.com/:x:/p/supreet_kumar/ETXL4GxQxDNBh_6d2Tt1MegBVVwQWplJz65zE61WvdhCvw", "Captured Logic Sheet")

# Add a dependency

data = {
            "type": {"name": "Depends" }, # Depends, Blocks
            "inwardIssue": { "key": edge['JiraTicketID_TO']}, #To if a dependency, # From if a blocker
            "outwardIssue": {"key": edge['JiraTicketID']} # From if a dependency, To if a blocker
            #"comment": { "body": "Linked related issue!"}
    }
jira.create_issue_link(data)

# Update issue status
jira.set_issue_status(issue_key,'Done')


# Update Priority (or any field)- PASS AS A PYTHON DICT NOT A JSON STRING :)
jira.update_issue_field('DATA-447', fields={'priority':{'name':'Critical'}})

# Update labels
myNewFields = {'labels': row['Labels']}                     
jira.update_issue_field('DATA-1', fields=myNewFields)

# Update description
jira.update_issue_field(key, fields={"description":new_desc})
    