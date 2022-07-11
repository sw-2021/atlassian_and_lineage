import pickle
import re

def Standardise_SQL_file (scriptstring):
    # This function is designed to remove some of the variability in spacing and newline formatting in a SQL script
    # It returns a list of semi-colon delimited, comments removed, spacing-normalised sql statements
    
    #### INPUTS
    # scriptstring is the text from a given .sql file
    
    #### Returns
    # subscripts is a list of individual SQL statements which have been cleaned up to facilitate regex etc. elsewhere
    #print(f'BEFORE: {scriptstring}')
    
    # Remove single line comments>> this has to happen BEFORE the collapsing of newline characters above
    string_clean=re.sub('--(.*)?(:?\n|$)','',scriptstring) # Remove code written behind comments --to end of line
    string_clean=re.sub('#(.*)?(:?\n|$)','',string_clean) # Remove code written behind comments --to end of line
    
    # REMOVE MULTIPLE SPACES AND NEWLINE INDICATORS
    string_clean=" ".join(string_clean.replace('\n',' ').split())# Remove newline characters and multiple spaces to standardise code
                          
    # REMOVE multi-line COMMENTS>> this has to happen AFTER the collapsing of newline characters above
    string_clean=re.sub('\/\*(.*?)\*\/','',string_clean) # Remove code written behind comments in /* */
    
    #print(f'AFTER: {string_clean}')
    
    
    #  PARSE INTO SEPERATE SCRIPTS
    
    # Replace quoted semi colons with a placeholder so that they aren't delimited when parsing SQL statements.
    # It doesn't solve for all occurences but reduces the likelihood of an error. The thing is we can't just assume a semi colon in quotes is wrong because of EXECUTE IMMEDIATE begin commonly used too
    
    #string_clean=string_clean.replace('";"','REPLACEMEWITHASEMICOLONANDDOUBLEQUOTES')
    #string_clean=string_clean.replace("';'",'REPLACEMEWITHASEMICOLONANDSINGLEQUOTES')
    string_clean=re.sub('"\s?;\s?"','REPLACEMEWITHASEMICOLONANDDOUBLEQUOTES',string_clean)
    string_clean=re.sub("'\s?;\s?'",'REPLACEMEWITHASEMICOLONANDDOUBLEQUOTES',string_clean)
    
    # Create subscripts where the data is split into different blocks of code using semicolons
    subscripts=string_clean.split(";") 
    
    # Now that subscripts have been derived, reverse the code from a line or two ago 
    subscripts=[i.replace('REPLACEMEWITHASEMICOLONANDDOUBLEQUOTES','";"') for i in subscripts]
    subscripts=[i.replace('REPLACEMEWITHASEMICOLONANDSINGLEQUOTES',"';'") for i in subscripts]
    
    return subscripts


def extract_tables_and_views_from_SQL_statement(script):
    
    #### INPUTS
    # script is a single SQL statement
    
    #### Returns 
    # from_tables and to_tables>> the tables that the script reads from and/or writes to in the SQL script
    
    from_tables=[]
    to_tables=[]
    
    #print(f'''EVALUATING SCRIPT:
    #      
    #      {script}''')
    
    
    ################################################################################################################################
    # First loop through subqueries by identifying all paired brackets
    # Keep only those in format (select ....) i.e. exclude the multitude of other bracket uses
    ################################################################################################################################
    
    bracket_opens=[] # Holds list of open bracket positions
    bracket_ranges=[] # Holds ranges for open and closed brackets

    # List all position of matches brackets>> NB might fall over if there is an open bracket in a quoted string somewhere...
    for pos in range(len(script)):
        if script[pos]=="(":
            bracket_opens.append(pos)
        elif script[pos]==")":
            try:
                bracket_ranges.append((bracket_opens[-1],pos)) # Add the matched pair to the pos list
                del bracket_opens[-1] # Remove the open bracket from the list
            except:
                print(f'''Found a closed bracket with an unmatched open bracket at character pos {pos} in
                     {script}''')

    for pair in bracket_ranges:
        subQ=script[pair[0]+1:pair[1]]
        
        # Pull out the select statements
        if subQ.lower().lstrip()[:6]=='select':
            #print('')
            #print(subQ)
            res=extract_tables_and_views_from_SQL_statement(subQ)
            #print(res)
            
            for from_tab in res[0]:
                from_tables.append(from_tab)
            for to_tab in res[1]:
                to_tables.append(to_tab)

    ################################################################################################################################
    # Create table statements
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language
    ################################################################################################################################
    
    create_table_result=re.findall('create? (?:or replace\s)?(temporary|temp|transient)?\s?table (?:if not exists\s)?([A-Za-z0-9_\-`\.]*?)[\s|\(]',script,re.IGNORECASE) 
    #print(f'CREATE TABLE RESULTS:{create_table_result}')
    for match in create_table_result:
        if len(match[1])>0:
            typeFlag='Temp Table' if len(match[0])>0 else 'Table'
            # If it's a temporary table, add "temp_data" as a dataset so that it is structured like data we want and the chain isn't broken
            #name='temp_data.'+match[1].replace('`','') if typeFlag=='Temp Table' else match[1].replace('`','') # Remove backticks from name
            name=match[1].replace('`','') # Remove backticks from name
            to_tables.append({'Name': name 
                              ,'Type':typeFlag
                             })
        
    ################################################################################################################################
    # CREATE VIEWS statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_view_statement
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_materialized_view_statement
    ################################################################################################################################
    create_view_result=re.findall('create (?:or replace\s)?(materialized)?\s?view (?:if not exists\s)?([A-Za-z0-9_\-`\.]*?)[\s|\(]',script,re.IGNORECASE) 
    #print(f'CREATE VIEW RESULTS:{create_table_result}')
    for match in create_view_result:
        if len(match[1])>0:
            typeFlag='Materialised View' if len(match[0])>0 else 'View'
            to_tables.append({'Name':match[1].replace('`','') # Remove backticks from name
                              ,'Type':typeFlag
                             })
    ################################################################################################################################
    # INSERT statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#insert_statement
    ################################################################################################################################
    insert_result=re.findall('insert (?:into\s)?([A-Za-z0-9_\-`\.]*?)\s?(?:values\s)?\(',script,re.IGNORECASE)
    #print(f'INSERT RESULTS:{insert_result}')
    for match in insert_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
        
    
    insert_result2=re.findall('insert (?:into\s)?([A-Za-z0-9_\-`\.]*?) (?:select|with)',script,re.IGNORECASE)
    for match in insert_result2:
        
        #print(f'Insert match found on {script}')
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
    ################################################################################################################################
    # DELETE statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#delete_statement
    ################################################################################################################################
    delete_result=re.findall('delete (?:from\s)?([A-Za-z0-9_\-`\.]*?)\s',script,re.IGNORECASE) 
    #print(f'DELETE RESULTS:{delete_result}')
    for match in delete_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
    ################################################################################################################################
    # TRUNCATE statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#truncate_table_statement
    ################################################################################################################################
    truncate_result=re.findall('truncate table ([A-Za-z0-9_\-`\.]*?)(?:\s|$|,)',script,re.IGNORECASE) 
    for match in truncate_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
    ################################################################################################################################
    # UPDATE statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#update_statement
    ################################################################################################################################
    update_result=re.findall('update ([A-Za-z0-9_\-`\.]*?)\s',script,re.IGNORECASE) 
    for match in update_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
    ################################################################################################################################
    # MERGE statements
    #https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#update_statement
    ################################################################################################################################
    merge_result=re.findall('merge (?:into\s)?([A-Za-z0-9_\-`\.]*?)\s',script,re.IGNORECASE) 
    for match in merge_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })

        
    ################################################################################################################################
    # FROM statements
    ################################################################################################################################
    script_list=script.split('UNION') # where there are multiple select statements, split them up
    for subQ in script_list:
        from_result=re.findall('select .*? from ([@A-Za-z0-9_\-`\.\*\/]*?)(?:\s|$|,|\()',subQ,re.IGNORECASE) 
        for match in from_result:
            from_tables.append({'Name':match.replace('`','') # Remove backticks from name
                             })
            
            
        #print(f'from RESULTS:{from_result}')
    ################################################################################################################################
    # JOIN statements
    ################################################################################################################################
    join_result=re.findall('join ([A-Za-z0-9_\-`\.\*]*?)(?:\s|$|,)',script,re.IGNORECASE) 
    #print(f'join RESULTS:{join_result}')
    for match in join_result:
        from_tables.append({'Name':match.replace('`','') # Remove backticks from name
                     })
    ################################################################################################################################
    # USING statements
    ################################################################################################################################
    using_result=re.findall('using ([A-Za-z0-9_\-`\.\*]*?)\s',script,re.IGNORECASE) # Does not pick up USING() to specify joins because this has brackets
    for match in using_result:
        from_tables.append({'Name':match.replace('`','') # Remove backticks from name
                     }) 
    ################################################################################################################################
    # CROSS JOINS listed with commas statements ~ Probably far from perfect but hopefully picks up most common syntax
    ################################################################################################################################
    xjoin_results=re.findall('select .*? (?:from|join)? [A-Za-z0-9_\-`\.\*]*?\s?,\s?([A-Za-z0-9_\-`\.\*]*?)(?:\s|$|,)',script,re.IGNORECASE) 
    #print(f'Xjoin RESULTS:{xjoin_results}')
    for match in xjoin_results:
        from_tables.append({'Name':match.replace('`','') # Remove backticks from name
                     }) 
        
    ################################################################################################################################
    # Pipes with a copy
    ################################################################################################################################
    #pipe_result=re.findall('create? (?:or replace\s)?pipe (?:if not exists\s)?([A-Za-z0-9_\-`\.]*?)\s.*copy into ([@A-Za-z0-9_\-`\.]*?)\s',script,re.IGNORECASE) 
    copy_into_result=re.findall('copy into ([@A-Za-z0-9_\-`\.]*?)\sfrom',script,re.IGNORECASE) 
    
    for match in copy_into_result:
        to_tables.append({'Name':match.replace('`','') # Remove backticks from name
                     }) 
       
    ################################################################################################################################
    # Streams
    ################################################################################################################################
    stream_results=re.findall('create? (?:or replace\s)?stream (?:if not exists\s)?([A-Za-z0-9_\-`\.]*?)\s(?:copy grants\s)?on\s?(?:external\s)?table\s?(.*?)(?:\s|$|,)',script,re.IGNORECASE)
    if len(stream_results)>0:
        for match in stream_results:
            from_tables.append({'Name':match[1].replace('`','') # Remove backticks from name
                     })
            to_tables.append({'Name':match[0].replace('`','') # Remove backticks from name
                              ,'Type':'stream'
                     })
        #    print(script)
#    print(from_tables,to_tables)
#    print('''
#    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    ''')



# RANDOM SPECIFIC PIECE OF QA
#     for x in from_tables:
#         for key,val in x.items():
#             #print(val)
#             if val=='transform.audits_hist_src':
#                 print('''
#                 ''')
#                 print(script)
#     for x in to_tables:
#         for key,val in x.items():
#             if val=='transform.audits_hist_src':
#                 print('''
#                 ''')
#                 print(script)

    return from_tables, to_tables
        
def process_SQL_file(file_item,Gpackage,skip_details=True):
    
    node_list,edge_list,node_details=Gpackage # Pull them out at this level, having passed from outer loop
    
    # Format them nicely, and split into individual SQL statements as delimited with semicolon
    sql_statements=Standardise_SQL_file(file_item) 

    # Loop through each script and extract table names
    for statement in sql_statements:

        # Extract a list of from tables and to tables
        from_list_statement,to_list_statement=extract_tables_and_views_from_SQL_statement(statement)
            
        #print(from_list_statement, to_list_statement)
        Add_EDGES=True 


        ############################################################################################################
        ### Process to_tables, if populated
        if len(to_list_statement)>0:
            
            # Append useful FILE attributes onto each individual SQL statement in a given file
            # List comprehension just filled list with none so I've avoided here
            for d in to_list_statement:
                #if skip_details==False:
                #    d.update({'RepoName':'snowflake_WH'#file_item.repository.name,
                #                       'ScriptName':file_item.name,
                #                       'ScriptPath':file_item.path,
                #                        'URL':file_item.html_url,
                #                       'DownloadURL':file_url,
                #                        'FileClass':file_item})
                    # Add the detailed info about each node
                #    node_details.append(d)
                # Add all the from and to node names to the more comprehensive node list (there might be some values missing detailed info e.g. tables created by other departments)
                node_list.append(d['Name'])

        else:
            Add_EDGES=False

        ############################################################################################################
        ### Process from_tables, if populated
        if len(from_list_statement)>0:
            for d in from_list_statement:
                # Add all the from and to node names to the more comprehensive node list (there might be some values missing detailed info e.g. tables created by other departments)
                node_list.append(d['Name'])

        else:
            Add_EDGES=False

        ############################################################################################################
        # Add edges to edgelist
        if Add_EDGES==True:
#             print('''
#                         EDGES ARE:''')
            for frm in from_list_statement:
                for to in to_list_statement:
                    #print('      ',frm['Name'],to['Name'])
                    edge_list.append((frm['Name'],to['Name']))
        
        
        
#        else:
#             print(statement)
#             print('''
#             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             ''')

