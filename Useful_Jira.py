import pandas as pd
import json

pd.options.display.max_rows = 999
import numpy as np
import re
import pickle
from atlassian import Jira
from pprint import pprint



def myprojectAndBoard(jira,projectName,boardnum):
    for i in jira.projects():
        if i['name'].strip()==projectName:
            myproj=i
            boards=jira.get_all_agile_boards(project_key=myproj['id'])
            myboard=[i for i in boards['values'] if i['id']==boardnum][0]
            return myproj,myboard
        
def get_succint_ticket_list(mytickets):
#     return [{'id':i['id']
# ,'key':i['key']
# ,'title': i['fields']['summary']
# ,'assignee_name':i['fields']['assignee'] 
# ,'status_name':i['fields']['status']['name']
# ,'status_id':i['fields']['status'] ['id']
# ,'issuetype_name':i['fields']['issuetype']['name']
# ,'issuetype_id':i['fields']['issuetype'] ['id']
# ,'project_id': i['fields']['project'] ['id']
# ,'project_key': i['fields']['project'] ['key']
# ,'project_name': i['fields']['project'] ['name']
# ,'labels': i['fields']['labels']
# ,'subtasks': i['fields']['subtasks']
# ,'parent_id': i['fields']['parent']['id']           
# ,'parent_key': i['fields']['parent']['key']
# ,'parent_title':i['fields']['parent']['fields']['summary']            
# ,'parent_type': i['fields']['parent']['fields']['issuetype']['name']           
# ,'parent_type_id': i['fields']['parent']['fields']['issuetype']['id']    
#             } if 'parent' in i['fields'].keys() else {'id':i['id']
# ,'key':i['key']
# ,'title': i['fields']['summary']
# ,'assignee_name':i['fields']['assignee'] 
# ,'status_name':i['fields']['status']['name']
# ,'status_id':i['fields']['status'] ['id']
# ,'issuetype_name':i['fields']['issuetype']['name']
# ,'issuetype_id':i['fields']['issuetype'] ['id']
# ,'project_id': i['fields']['project'] ['id']
# ,'project_key': i['fields']['project'] ['key']
# ,'project_name': i['fields']['project'] ['name']
# ,'labels': i['fields']['labels']
# ,'subtasks': i['fields']['subtasks']    
#             } 
#             for i in mytickets ]

    ticket_list=[]
    for i in mytickets:
        
    # Core fields
        x={'id':i['id']
    ,'key':i['key']
    ,'title': i['fields']['summary']
    ,'assignee_name':i['fields']['assignee'] 
    ,'status_name':i['fields']['status']['name']
    ,'status_id':i['fields']['status'] ['id']
    ,'issuetype_name':i['fields']['issuetype']['name']
    ,'issuetype_id':i['fields']['issuetype'] ['id']
    ,'project_id': i['fields']['project'] ['id']
    ,'project_key': i['fields']['project'] ['key']
    ,'project_name': i['fields']['project'] ['name']
    ,'labels': i['fields']['labels']
    ,'subtasks': i['fields']['subtasks'] 
    ,'completion_date':i['fields']['resolutiondate']
                }

    # Only add parent where relevant
        if 'parent' in i['fields'].keys():
            x={**x,**{'parent_id': i['fields']['parent']['id']           
    ,'parent_key': i['fields']['parent']['key']
    ,'parent_title':i['fields']['parent']['fields']['summary']            
    ,'parent_type': i['fields']['parent']['fields']['issuetype']['name']           
    ,'parent_type_id': i['fields']['parent']['fields']['issuetype']['id']}}


    # Only add description where exists    
        if 'description' in  i['fields'].keys():
            x={**x,**{'description':i['fields']['description']}}

        else:
            x={**x,**{'description':i['fields']['summary']}} # Else re-use summary info

        ticket_list.append(x)
    return ticket_list


def get_epics_only(mytickets):
    board_epics=[]
    for i in mytickets:
        if 'fields' in i.keys():    
            try:
                if i['fields']['issuetype']['name']=='Epic':
                    board_epics.append({'Epic Name':i['fields']['summary'], 'Epic Key':i['key']})
            except:
                pass
        elif 'issuetype_name' in i.keys():
            try:
                if i['issuetype_name']=='Epic':
                    board_epics.append({'Epic Name':i['title'], 'Epic Key':i['key']}
                                       )
            except:
                pass 
    epic_name_list=[i['Epic Name'] for i in board_epics]
    epic_key_list=[i['Epic Key'] for i in board_epics]
    board_epic_key_lookup={i['Epic Name']:i['Epic Key'] for i in board_epics}

    return board_epics,epic_name_list, epic_key_list,board_epic_key_lookup



def get_all_project_issues_uncapped(jira,proj_key,field_list=None):    
    queryme=True
    start_idx=0
    mycards=[]

    if field_list:
        while queryme==True:
            myres=jira.get_all_project_issues(proj_key,fields=field_list,start=start_idx,limit=start_idx+100)
            if len(myres)==0:
                queryme=False
            else:
                # Append the results
                mycards=mycards+myres
                # Update the start number
                start_idx+=100
    else:
        while queryme==True:
            myres=jira.get_all_project_issues(proj_key,start=start_idx,limit=start_idx+100)
            if len(myres)==0:
                queryme=False
            else:
                # Append the results
                mycards=mycards+myres
                # Update the start number
                start_idx+=100
    return mycards



def get_epic_of_nested_cards(mytickets_succint):
    """ Requires the 'mytickets_succinct' parameter'
    Returns: a df listing the JIRA key mapped to the JIRA name and key of the epic that a task or subtask eventually resolves to, if any'
    """
    # Create a df just showing parent info
    parent_df=pd.DataFrame(mytickets_succint).loc[:,['key','issuetype_name','parent_key','parent_type','parent_title']]
    # Filter out those without a parent (orphaned cards or Epics themselves)
    parent_df=parent_df.loc[parent_df['parent_key'].notna()]
    
    
    num_nested_cards=len(parent_df.loc[parent_df['parent_type']!='Epic'])


    while num_nested_cards>0:
        print(num_nested_cards)
        # Merge on parent's parent info
        new_parent_df=parent_df.merge(parent_df,how='left', left_on='parent_key',right_on='key',suffixes=['','_of_parent'])

        # Where parent not already an epic, update
        new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_key']=new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_key_of_parent']
        new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_title']=new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_title_of_parent']
        # Update type last, as this is used to filter...
        new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_type']=new_parent_df.loc[new_parent_df['parent_type']!='Epic','parent_type_of_parent']

        # Rebuild parent_df
        parent_df=new_parent_df.loc[new_parent_df['parent_key'].notna(),['key','issuetype_name','parent_key','parent_type','parent_title']]
        num_nested_cards=len(parent_df.loc[parent_df['parent_type']!='Epic'])
    epic_df=parent_df.rename(columns={'parent_key':'epic_key','parent_type':'epic_type','parent_title':'epic_title'}).loc[:,['key','epic_key','epic_title']]
    
   
    return epic_df



def jira_linked_issues_as_edges(tickets,tickets_succinct,ticket_keys=[],explode_subtasks=True):

    edges=[]

    #################################################################
    # Edges as explicitly defined in Jira dependencies- blockers and dependencies
    #################################################################
    # Loop tickets
    for x in tickets:

        # If the ticket has links
        if 'issuelinks' in x['fields'].keys():

            # Loop through them
            for y in x['fields']['issuelinks']:

                # If they are dependencies
                if y['type']['name'] in ['Depends']:

                    # 'name': 'Depends', 'inward': 'is depended on by', 'outward': 'depends upon', 'self': 'https://sedexsolutions.atlassian.net/rest/api/2/issueLinkType/10400'}


                    # If an outward issue, current ticket depends upon "outward" ticket.
                    if 'outwardIssue' in y.keys():
                        edges.append((y['outwardIssue']['key'],x['key']))

                    # If an inward issue, current ticket is depended upon by "inward" ticket.
                    if 'inwardIssue' in y.keys():
                        edges.append((x['key'],y['inwardIssue']['key']))



                if y['type']['name'] in ['Blocks']:

                    #'name': 'Blocks', 'inward': 'is blocked by', 'outward': 'blocks'


                    # If an outward issue, current ticket depends upon "outward" ticket.
                    if 'outwardIssue' in y.keys():
                        edges.append((x['key'],y['outwardIssue']['key']))

                    # If an inward issue, current ticket is depended upon by "inward" ticket.
                    if 'inwardIssue' in y.keys():
                        edges.append((y['inwardIssue']['key'],x['key']))


    edges=list(set(edges))
    
    #################################################################
    # Edges between subtasks and a parent
    #################################################################
    if explode_subtasks==True:
        edges_subtasks=[]
        # For each card
        for x in tickets_succinct:

            # If the card has subtasks
            if 'subtasks' in x.keys():
                if len(x['subtasks'])>0:

                    # For each of those subtasks, create a dependency on the subtask being complete before the parent ticket is.
                    for y in x['subtasks']:
                        edges_subtasks.append((y['key'],x['key']))

        # De dupe
        edges_subtasks=list(set(edges_subtasks))
        
        # Add them together
        all_edges=list(set(edges+edges_subtasks))

    # If a validation list e.g. of known nodes was passed, cross reference against this list.
    if len(ticket_keys)>0:
        all_edges=[i for i in all_edges if i[0] in ticket_keys and i[1] in ticket_keys]
    
    return all_edges