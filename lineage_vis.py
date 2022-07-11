import re
import pickle
import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from pyvis.network import Network 

import collections.abc
import json
from pprint import pprint

def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def newline_every_n(instr,step=3):
    ### This function is a simple one to parse a long string into a newline delimited every few words
    spacecount=0
    outstr=''
    for n,char in enumerate(instr):
        if char.isspace():
            spacecount+=1
            if spacecount%step==0:
                outstr+='\n'
            else:
                outstr+=char
        else:
            outstr+=char
    return outstr


def check_is_DAG(G):
    if nx.is_directed_acyclic_graph(G):
        print("We have a directed acyclic graph - continue")
        True
    else : 
        cycle = nx.find_cycle(G)
        print("Uhoh there is at least one cycle - you need to remove them")
        print(cycle)
        return False
    
def shift_successors_right(node,G,xposDict):
    # List a nodes successors
    succ=list(G.successors(node))
    if len(succ)==0:
        return
    else:
        for s in succ: # For each successor/ dependent
            if xposDict[s]<=xposDict[node]: # If successor is positionally equal to or left of the dependant node
                xposDict[s]=xposDict[node]+1 # Update that node to be one to the right
                #print('Updating {} and searching for successors'.format(s))
                shift_successors_right(s,G,xposDict) # Then loop through the successor's successors
    return None

def shift_ancestors_left(node,G,xposDict):
    # List a nodes successors
    ancs=list(G.predecessors(node))
    if len(ancs)==0:
        return
    else:
        for a in ancs: # For each successor/ dependent
            if xposDict[a]>=xposDict[node]: # If successor is positionally equal to or left of the dependant node
                xposDict[a]=xposDict[node]-1 # Update that node to be one to the right
                #print('Updating {} and searching for successors'.format(s))
                shift_ancestors_left(a,G,xposDict) # Then loop through the successor's successors
    return None


    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vis1_graphDescriptiveStats(node_df,G,showtiming=False):
    #########################################################################################################################
    # Set up Timing 
    #########################################################################################################################
    start_time = time. time()
    
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return
    
    #########################################################################################################################
    # Generate Actual Descriptive Stats as per NetWorkX capabilities
    #########################################################################################################################
    ###################################################
    # Node classification as start, end, interim
    ###################################################
    def classify_node(node):
        if G.in_degree()[node]==0 and G.out_degree()[node]==0:
            nodeType='unconnected'
        elif G.in_degree()[node]==0 and G.out_degree()[node]>0:
            nodeType='start'
        elif G.in_degree()[node]>0 and G.out_degree()[node]==0:
            nodeType='terminal'
        else:
            nodeType='interim'
        return nodeType
    
    # Classify nodes according to connectivity
    #node_df['classification']=node_df.apply(lambda x: classify_node(x['name']),axis=1)
    node_df['classification']=np.vectorize(classify_node, otypes=[np.str])(node_df['name'])
    

    ###################################################
    # Degree Centrality (proportion of nodes it is connected to)
    ###################################################
    degree_centrality_dict=nx.degree_centrality(G) # dictionary of each node's degree centrality
    #node_df['degree_centrality']=node_df['name'].apply(lambda x: degree_centrality_dict[x])
    node_df['degree_centrality']=np.vectorize(degree_centrality_dict.get)(node_df['name'])
    
    ###################################################
    # Betweenness Centrality
    ###################################################
    betweenness_centrality_dict=nx.betweenness_centrality(G) # dictionary of each node's betweenness centrality
    #node_df['betweenness_centrality']=node_df['name'].apply(lambda x: betweenness_centrality_dict[x])
    node_df['betweenness_centrality']=np.vectorize(betweenness_centrality_dict.get)(node_df['name'])
    
    ###################################################
    # Node successors
    ###################################################
    def vct_successors(x):
        return list(G.successors(x))
    #node_df['successors']=node_df['name'].apply(lambda x: list(G.successors(x)))
    node_df['successors']=np.vectorize(vct_successors, otypes=[np.ndarray])(node_df['name'])
    
    ###################################################
    # Node predecessors
    ###################################################
    def vct_predecessors(x):
        return list(G.predecessors(x))
    #node_df['predecessors']=node_df['name'].apply(lambda x: list(G.predecessors(x)))
    node_df['predecessors']=np.vectorize(vct_predecessors, otypes=[np.ndarray])(node_df['name'])
    
    ###################################################
    # Node Neighbours
    ###################################################
    node_df['all_neighbours']=node_df.apply(lambda x: x['successors']+x['predecessors'],axis=1)
        
    print(f'After getting some node attributes, node_df is {len(node_df)} long')
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Generation of graph attributes took: {time_elapsed}')
    return node_df
        
        
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vis2_validate_dfs(node_df,G,edge_df
                      ,def_color='black'
                      ,def_shape='square'
                      ,def_size=1
                      ,def_yband='~~~'
                      ,def_edgeColor='grey'
                      ,def_edgeSize=1
                      ,def_edgeText=''
                      ,showtiming=False):
    
    #########################################################################################################################
    # Set up Timing 
    #########################################################################################################################
    start_time = time. time()
    
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that edge_df is a DataFrame
    if not isinstance(edge_df, pd.DataFrame):
        print('ERROR: cannot continue- edge_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return
    
    
    ############################ Validate fields in node_df ############################
    # Validate Name property
    if 'name' not in node_df.keys():
        print(f'No name field present in node_df. Please pass a field called "name" to identify nodes')
        return
    
    # Validate color property
    if 'color' not in node_df.keys():
        print(f'No colour field specified in node_df. Default value of "{def_color}" will be used')
        node_df['color']=def_color
    else:
        node_df['color']=node_df['color'].fillna(def_color) # Fill in the blanks

    # Validate shape property
    if 'shape' not in node_df.keys():
        print(f'No shape field specified in node_df. Default value of "{def_shape}" will be used')
        node_df['shape']=def_shape
    else:
        node_df['shape']=node_df['shape'].fillna(def_shape) # Fill in the blanks

    # Validate size property
    if 'size' not in node_df.keys():
        print(f'No size field specified in node_df. Default value of "{def_size}" will be used')
        node_df['size']=def_size
    else:
        node_df['size']=node_df['size'].fillna(def_size) # Fill in the blanks
    
    # Validate y band property
    if 'y_band' not in node_df.keys():
        print(f'No y band grouping specified. Default value of "{def_yband}" will be used')
        node_df['y_band']=def_yband # Leave blank if
    else:
        node_df['y_band']=node_df['y_band'].fillna('Other') # Fill in the blanks with "Other"
        
    # Validate text property
    if 'text' not in node_df.keys():
        print(f'No text field passed to node_df. The node name will be used by default')
        node_df['text']=node_df['name'] # Leave blank if
    else:
        node_df['text']=np.where(node_df.text.isnull(), node_df.name, node_df.text)

    # Validate label property
    if 'label' not in node_df.keys():
        print(f'No label field passed to node_df. The node name will be used by default')
        node_df['label']=node_df['name'] # Leave blank if
    else:
        node_df['label']=np.where(node_df.label.isnull(), node_df.name, node_df.label)

    # Image URLs
    if 'imageURL' not in node_df.keys():
        print('Warning: there is no image URL field. Pyvis layouts needing images will fail')
        node_df['imageURL']=None
        
    # graph stats
    for col in ['classification','degree_centrality','betweenness_centrality','successors','predecessors','all_neighbours']:
        if col not in node_df.keys():
            print(f'''Warning: node_df is missing the field {col}- this might prevent the visualisation generation from working.
                  Consider running function lv.vis1_graphDescriptiveStats to generate these fields.''')
            
    
    ############################ Validate fields in edge_df ############################
    
    # Validate Name property
    if 'To' not in edge_df.keys():
        print(f'No "To" field present in edge_df. Please pass a field called "To" to identify nodes')
        return

    # Validate Name property
    if 'From' not in edge_df.keys():
        print(f'No "From" field present in edge_df. Please pass a field called "From" to identify nodes')
        return
    
    
    if 'color' not in edge_df.keys():
        print(f'''No colour field specified in edge_df. Default value of "{def_edgeColor}" will be used.
              Please pass a field called "size" to manage''')
        edge_df['color']=def_edgeColor
    else:
        edge_df['color']=edge_df['color'].fillna(def_edgeColor) # Fill in the blanks

        
    if 'size' not in edge_df.keys():
        print(f'''No size/line weight field specified in edge_df. Default value of "{def_edgeSize}" will be used.
              Please pass a field called "size" to manage''')
        edge_df['size']=def_edgeSize
    else:
        edge_df['size']=edge_df['size'].fillna(def_edgeSize) # Fill in the blanks   
    
    if 'text' not in edge_df.keys():
        print(f'''No text field specified in edge_df to annotate edges. Default value of "{def_edgeText}" will be used.
              Please pass a field called "size" to manage''')
        edge_df['text']=def_edgeText
    else:
        edge_df['text']=edge_df['text'].fillna(def_edgeText) # Fill in the blanks   
    
    
    # Validate label property
    if 'label' not in edge_df.keys():
        print(f'No label field passed to edge_df. It will be blank by default')
        edge_df['label']='' # Leave blank if
    else:
        edge_df['label']=np.where(edge_df.label.isnull(), '', edge_df.label)
    
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Validation of objects took: {time_elapsed}')
    return node_df,edge_df








# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vis3_calc_xpos(G, node_df,showtiming=False,x_fill_direction='l_to_r'):
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return
    
    # Check that node_df contains all the columns it needs to 
    required_columns=['y_band','classification','name'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in node_df.columns:
            print(f'Function requires a field called "{col}". Please consider running vis1_graphDescriptiveStats and/or vis2_validate_dfs to generate them')
            exit_flag=True
    if exit_flag==True:
        return
    
    if 'xpos' in node_df.columns:
        node_df=node_df.drop(columns=['xpos'])
        
    # Check that node_df the same nodes as G
    if len(set(G.nodes()))!=len(set(node_df['name'])):
        print('There is a mismatch between the number of nodes in the Graph object G, and the node_df')
        return
    elif len(set(G.nodes()))!=len(set(list(G.nodes())+list(node_df['name']))):
        print('There is a mismatch between the node names in the Graph object G and the node_Df')
        return
    
    
    #########################################################################################################################
    # Set up Timing 
    #########################################################################################################################
    start_time = time. time()
    
    ##############################################################################################
    # Next, calc xpos. This is an absolute value. The scaling between 0 and 1 will happen later
    ##############################################################################################
    # Initialise all node x positions as 0
    xposDict={node:0 for node in G.nodes()}

    # Loop all nodes and shift their successors right of them
    for node in G.nodes():
        if x_fill_direction=='l_to_r':
            shift_successors_right(node,G,xposDict) # Call to function defined earlier
        else:
            shift_ancestors_left(node,G,xposDict)
            
    # Print Timings
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Shifting of successor xpositions took: {time_elapsed}')
        start_time = time. time()
    
    # Store the positions in a dataframe
    xposdf=pd.DataFrame.from_dict(xposDict,orient='index').reset_index().rename(columns={0:'xpos','index':'name'}) # Store x positions in a dataframe
    node_df=node_df.merge(xposdf,how='inner',on='name') # Join them in
    max_x=node_df['xpos'].max() # Identify the maximum positional width
    

    #### Update the xposition of unconnected nodes to just span the range
    # Identify the Y-banding groups
    Grouplist=list(node_df['y_band'].unique())
              
              
    for Group in Grouplist: #For each y band
        # List unconnected nodes in the group
        unconnected_nodes=node_df[(node_df['classification']=='unconnected')&(node_df['y_band']==Group)]['name']
        
        # Assign them an x pos across the range
        unconnected_pos={node:int(n%(max_x+1)) for n,node in enumerate(unconnected_nodes)} 
        unconnected_nodes=pd.DataFrame(unconnected_nodes,columns=['name'])
        unconnected_nodes['xpos']=unconnected_nodes['name'].apply(lambda x: unconnected_pos[x])
        
        # Join it back in
        node_df=node_df.merge(unconnected_nodes,how='left',on='name',suffixes=[None,"_new"])
        
        # Update the column
        node_df['xpos'] = np.where(node_df["xpos_new"].isnull(), node_df["xpos"], node_df["xpos_new"] )
        
        # Drop the added column
        node_df=node_df.drop(columns=['xpos_new'])
    
    # For r_to_l it can create negative values, which we need to shift over.
    min_x=node_df['xpos'].min()
    if min_x<0:
        node_df['xpos']=node_df['xpos']+abs(min_x)
    
        # Print Timings
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'And generating xpositions for unconnected nodes took: {time_elapsed}')
        start_time = time. time()
    
    print(f'After setting xpos, node_df is {len(node_df)} long')
    return node_df









# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vis4_calc_ypos(G, node_df,showtiming=False ,spacing_size=1
                               ,margins=0.3
                               ,fill_from='l_to_r'
                                , print_update_every=20):
   
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return

    # Check that node_df contains all the columns it needs to 
    required_columns=['name','y_band','xpos'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in node_df.columns:
            print(f'Function requires a field called "{col}". Please consider running vis1_graphDescriptiveStats, vis2_validate_dfs and/or vis3_calc_xpos to generate them')
            exit_flag=True
    if exit_flag==True:
        return
    
    # Remove the columns generated by this procedure, in the event that it is re-run. It could (& does) cause conflicts
    generated_cols=['y_band_height','y_band_height_with_spacing', 'y_band_macro_order', 'y_band_boundary','y_band_From', 'y_band_To', 'y_band_Range', 'y_band_midpoint', 'ypos']
    
    keepcols=node_df.columns
    keepcols=[i for i in keepcols if i not in generated_cols]    
    node_df=node_df.loc[:,keepcols]

    # Check that node_df the same nodes as G
    if len(set(G.nodes()))!=len(set(node_df['name'])):
        print('There is a mismatch between the number of nodes in the Graph object G, and the node_df')
        return
    elif len(set(G.nodes()))!=len(set(list(G.nodes())+list(node_df['name']))):
        print('There is a mismatch between the node names in the Graph object G and the node_Df')
        return
    
    # Check that update_freq is an integer
    if not isinstance(print_update_every, int):
        print('ERROR: cannot continue- parameter "print_update_every" must be an integer!')
        return

    
    
    #########################################################################################################################
    # Make deep copies to manipulate in a ringfenced way.
    #########################################################################################################################
    nodeDf=node_df.copy(deep=True)
    #edgeDf=edge_df.copy(deep=True)
     
    #########################################################################################################################
    # Set up Timing 
    #########################################################################################################################
    start_time = time. time()
    
    #########################################################################################################################
    # Absolute y-band positions and possible range.
    #########################################################################################################################
    
    ## We want to organise such that edges between nodes in different bands don't have to go across the page. Therefore the overall
    ## ordering of the y-bands matters. This attempts to scope that order by identifying the most interconnected y-band, and then
    ## calculating how connected it is to other bands. They are then organised such that the most interconnected band is in the middle
    ## of the visualisation, the second most just above it, the third most just below it, etc.
    
  
    # Generate an edge list that instead maps y-band to y-band (rather than node name to node name)
    # NB the item() clause expects only one result, so the next couple of lines are error checking that ideally are never evoked if dupes are removed from nodeDf
    for edge in G.edges():
        if len(nodeDf.loc[nodeDf['name']==edge[0],'y_band'])!=1:
            print('The error you''re about to hit is because this result set is not 1 item long')
            print('Edge 0 is ',edge[0])
            print(nodeDf.loc[nodeDf['name']==edge[0],['name','y_band']])

        if len(nodeDf.loc[nodeDf['name']==edge[1],'y_band'])!=1:
            print('The error you''re about to hit is because this result set is not 1 item long')
            print('Edge 1 is ',edge[1])
            print(nodeDf.loc[nodeDf['name']==edge[1],['name','y_band']])
    
    band_edges=[(nodeDf.loc[nodeDf['name']==edge[0],'y_band'].item(),nodeDf.loc[nodeDf['name']==edge[1],'y_band'].item()) for edge in G.edges()]  
    # Remove self-connected edges, these don't cause overlaps
    
    # Store incase needed
    band_G_nodes=[]
    for edge in band_edges:
        if edge[0] not in band_G_nodes:
            band_G_nodes.append(edge[0])
        if edge[1] not in band_G_nodes:
            band_G_nodes.append(edge[0])
    
    # Remove self joins
    band_edges=[edge for edge in band_edges if edge[0]!=edge[1]]
    
    # If more than 1 interconnected band
    if len(band_edges)>0:
        # Turn it into a graph
        band_G=nx.Graph()
        band_G.add_edges_from(band_edges)

        # Calculate the centrality to see the bands with the most interconnectivity
        band_centrality_dict=dict(nx.degree_centrality(band_G))

        # Store the most connected band as an anchor point
        most_connected=max(band_centrality_dict, key=band_centrality_dict.get)
        # Find how similar the other bands are to this anchor
        band_closeness=nx.simrank_similarity(band_G,most_connected)
    else:
        band_closeness={node:1 for node in band_G_nodes}
    #print(f'The most connected band is {most_connected}. The similarity of every other band to this is {band_closeness}')
    
    # We want to put the most connected band in the middle of the y-axis, not at the top, so we need to calculate the midpoint as a function of how many bands there are
    # Identify the Y-banding groups
    Grouplist=list(nodeDf['y_band'].unique())
    
    midpoint=np.ceil(len(Grouplist)/2)
    
    tmpdf=pd.DataFrame(Grouplist,columns=['y_band']) # Hold all y bands, even if no overlapping edges, to left join onto
    
    # Calculate the order of each band from the middle to return a numerical ordering of bands
    band_order_df=pd.DataFrame.from_dict(band_closeness,orient='index',columns=['similarity']).reset_index()\
    .rename(columns={'index':'y_band'})
    
    band_order_df=tmpdf.merge(band_order_df, on='y_band',how='left').fillna(0)
    band_order_df=band_order_df.sort_values(by='similarity',ascending=False).reset_index(drop=True)
    band_order_df['order']=np.floor(band_order_df.index/2)
    band_order_df['sign']=band_order_df.index%2
    band_order_df['sign2']=((-1)**band_order_df['sign'])*(-1)
    band_order_df['y_band_macro_order']=midpoint+band_order_df['sign2']*band_order_df['order']+band_order_df['sign']
    # We'll just leave this here until we need it now ...
    
    
    print(f'After calculating ypos band positions, nodeDf is {len(nodeDf)} long')
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Calculating ypos band positions took: {time_elapsed}')
        start_time = time. time()
    
    #########################################################################################################################
    # Calculation of specific ypos
    #########################################################################################################################
    
    # Now, for each group, for each x position, work out how many nodes there will be in the same vertical column
    heights_by_group_and_xpos=nodeDf[['xpos','y_band','name']].groupby(['xpos','y_band']).count().reset_index()
    # For each group, work out what the maximum column height needs to be
    heights_by_group=heights_by_group_and_xpos[['y_band','name']].groupby('y_band').max()
    
    #display(heights_by_group)
    # For each group, work out what the maximum column height needs to be, and add spacing position between graphs
    heights_by_group['y_band_height_with_spacing']=heights_by_group['name']+spacing_size
    heights_by_group=heights_by_group.rename(columns={'name':'y_band_height'}).reset_index()
    #display(band_order_df[['y_band','y_band_macro_order']])
    # Add in the overall position of the y_band as calculated earlier
    heights_by_group=heights_by_group.merge(band_order_df[['y_band','y_band_macro_order']],on='y_band',how='left')
    # Order by the overall position of the y-band
    heights_by_group=heights_by_group.sort_values(by='y_band_macro_order')
    # Calculate the individual y-positions that span the range
    heights_by_group['y_band_boundary']=heights_by_group['y_band_height_with_spacing'].cumsum()
    heights_by_group['y_band_From']=heights_by_group['y_band_boundary'].shift(1).fillna(0)
    heights_by_group['y_band_To']=heights_by_group['y_band_From']+heights_by_group['y_band_height']
    # Generate a nested list of the range that items can theoretically take. This is your constraints
    heights_by_group['y_band_Range']=heights_by_group.apply(lambda x: list(range(int(x['y_band_From']),int(x['y_band_To'])+1)),axis=1)
    heights_by_group['y_band_midpoint']=heights_by_group['y_band_From']+np.ceil(heights_by_group['y_band_height']/2)
    
    
    # Merge the info back into nodeDf        
    nodeDf=nodeDf.merge(heights_by_group,how='inner',on='y_band')
    #display(nodeDf.head())
    
    print(f'After calculating yband heights, nodeDf is {len(nodeDf)} long')
    
    # Timing 
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Calculating ypos band heights took: {time_elapsed}')
        start_time = time. time()
    
    ##############################################################################################
    # Assign y-pos
    ##############################################################################################
    posDf=heights_by_group
    posDf['y_pos_Available']=posDf['y_band_Range'] # The 'available' variable holds unallocated spaces
    max_x=nodeDf['xpos'].max() # Identify the maximum positional width
    
    posDf['xpos']=posDf.apply(lambda x: list(range(0,int(max_x)+1)),axis=1)
    posDf=posDf.explode('y_pos_Available')
    posDf=posDf.explode('xpos')
    #display(posDf)

    # Initialise with longest path per group in the middle of the group
    for Group in Grouplist:
        subG_nodes=list(nodeDf.loc[nodeDf['y_band']==Group,'name']) # List the nodes within a group only
        subG=G.subgraph(subG_nodes) # Create a subgraph
        subG_longest_path=nx.dag_longest_path(subG) # Identify the longest path, this will be the main vein running through the y band
        
        for node in subG_longest_path:
            # Get the xpos and y_band of the node
            xpos=nodeDf.loc[nodeDf['name']==node,'xpos'].item() # Get the updated nodes x position
            #print(xpos)
            y_band=nodeDf.loc[nodeDf['name']==node,'y_band'].item() # Get the updated nodes y band
            #print(y_band)
            
            # Determine the available positions within that xpos and yband
            #available_positions=nodeDf.loc[(nodeDf['xpos']==xpos)&(nodeDf['y_band']==y_band),'y_pos_Available'].item()
            available_positions=list(posDf.loc[(posDf['xpos']==xpos)&(posDf['y_band']==y_band),'y_pos_Available'])
            #print(available_positions)
            
            # Calculate the best position for this node
            best_ypos=nodeDf.loc[nodeDf['name']==node,'y_band_midpoint'].item()
            #print(best_ypos)
            
            # Assign it to the best position
            nodeDf.loc[nodeDf['name']==node,'ypos']=best_ypos # Initialise on the midpoint
            
            
            # Remove available position from the list of available positions
            available_positions=[pos for pos in available_positions if pos!=best_ypos] # Remove the ypos from the list
            #print(available_positions)
            posDf=posDf.loc[~((posDf['xpos']==xpos)&(posDf['y_band']==y_band)*(posDf['y_pos_Available']==best_ypos)),] # And update the dataframe holding available y positions

    # Now use a greedy heuristic to map the rest of the nodes
    # We can fill in from left to right (source to terminus) or backwards from right to left
    
    print(f'After ypos initialisation, nodeDf is {len(nodeDf)} long')
    
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'ypos initialisation took: {time_elapsed}')
        start_time = time. time()
    
    ###############
    # This just generates nodes in an intelligent order to have ypos generated
    ###############
    search_order=[]
    
    if fill_from=='r_to_l':
        # Get all the "end" nodes, which would be on the right of the visual. Sort them by descending degree centrality so that we deal with the most interconnected nodes first
        start_from_nodes=nodeDf.loc[nodeDf['classification']=='terminal',['name','degree_centrality']].sort_values(by='degree_centrality',ascending=False) # List start nodes
        start_from_node_list=list(start_from_nodes['name'])
        
        # Create a list of every node on the graph and the order in which we should add a y-pos.
        for start_node in start_from_node_list: # For every start node
            search_order.append(start_node)
            for succ in list(nx.dfs_predecessors(G,start_node)): # And the successor of every start node
                if succ not in search_order:
                    search_order.append(succ) # Add to the search list if not already
    else: # Assume 'l_to_r'
        
        # Get all the "start" nodes, which would be on the left of the visual. Sort them by descending degree centrality so that we deal with the most interconnected nodes first
        start_from_nodes=nodeDf.loc[nodeDf['classification']=='start',['name','degree_centrality']].sort_values(by='degree_centrality',ascending=False) # List start nodes
        start_from_node_list=list(start_from_nodes['name'])
        
        # Create a list of every node on the graph and the order in which we should add a y-pos.
        
        for start_node in start_from_node_list: # For every start node
            search_order.append(start_node)
            for succ in list(nx.dfs_successors(G,start_node)): # And the successor of every start node
                # print(f'Successors are:{succ}')
                if succ not in search_order:
                    search_order.append(succ) # Add to the search list if not already
   
    # Mop up any missed. In theory this should only be unconnected nodes. Everything else should link to a source or terminal node, but who knows
    for node in G.nodes():
        if node not in search_order:
            search_order.append(node)
    
    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Ordering nodes in order to calculate their position took: {time_elapsed}')
        start_time = time. time()
    
    ###############
    # Now, iterate through all available node positions, picking that which minimises the average distance to a neighbour
    ###############
    total_n_nodes=len(search_order)
    print(f'nodeDf is {len(nodeDf)} long, and we are searching over {total_n_nodes}')

    for n,node in enumerate(search_order):
        if n%print_update_every==0:
            print(f'Generating y position for node {n} of {total_n_nodes}')

        # Get the current ypos, if any
        try:
            if nodeDf.loc[nodeDf['name']==node,'ypos'].isna().item():
                current_ypos=None
            else:
                current_ypos=nodeDf.loc[nodeDf['name']==node,'ypos'].item()
        except:
            print(node)
            print(nodeDf.loc[nodeDf['name']==node,'ypos'].isna())
            
        # Get the xpos and y_band of the node
        xpos=nodeDf.loc[nodeDf['name']==node,'xpos'].item() # Get the updated nodes x position
        y_band=nodeDf.loc[nodeDf['name']==node,'y_band'].item() # Get the updated nodes y band
 
        # Determine the available positions within that xpos and yband
        if n%print_update_every==0:
            avtime=time.time()
        available_positions=list(posDf.loc[(posDf['xpos']==xpos)&(posDf['y_band']==y_band),'y_pos_Available'])
        if n%print_update_every==0:
            avtime_elapsed = (time.time() - avtime)
            if showtiming==True:
                print(f'To calc available positions tooks: {avtime_elapsed}')
        #print(available_positions)

        # Calculate the best position for this node
        if n%print_update_every==0:
            bsttime=time.time()
        ######################################################################################################################################################################################################################################################################################################################################################################################
        ################# THIS IS THE SLOW LINE!!!!
#         distance_positions={i:Avg_distance_to_neighbours(node,i,nodeDf) for i in available_positions} # Get the average distance from neighbours    
        
#         # Looping each node, this takes half a second to loop through all available positions
        
#         #dict(zip(available_positions,np.vectorize(Avg_distance_to_neighbours)(node,i,nodeDf)
#         #distance_positions={i:Avg_distance_to_neighbours(node,i,nodeDf) for i in available_positions} # Get the average distance from neighbours    
#         if n%print_update_every==0:
#             bsttime_elapsed = (time.time() - bsttime)
#             print(f'To calc best positions tooks: {bsttime_elapsed}')
#         best_ypos=min(distance_positions, key=distance_positions.get) # Select the minimum distance
#         print(best_ypos)
        
        if n%print_update_every==0:
            bsttime2=time.time()
       
        posdf=pd.DataFrame(available_positions,columns=['Avail'])
        #print(len(posdf))
        # Create a df just mapping a node to it's neighbours- one neighbour per row
        neighbourdf=nodeDf.loc[(nodeDf['name']==node),['name','all_neighbours']].explode('all_neighbours')
        # Get neighbour positions, if they have any
        neighbourdf=pd.merge(neighbourdf,nodeDf.loc[nodeDf['ypos'].notna(),['name','ypos']], left_on='all_neighbours',right_on='name')
        
        if len(neighbourdf)>0: #If there are neighbours to the node, and if they already have positions applied
            ### THIS PART WILL STILL FAIL IF posdf is empty. This might occur when you've overriden the allocation of the x positioning in the step 3 function- leaving no feasible space
            #print(len(neighbourdf))
            # Cross join with available positions
            neighbourdf['j']=99
            posdf['j']=99
            neighbourdf=pd.merge(neighbourdf,posdf,left_on='j',right_on='j')

            # Calc Distance between each available position that the node could go in, and all the nodes neighbours
            # name=node,avail= potential position for node all_neighbours=neighbourname, , ypos= neighbour position
            neighbourdf['Distance']=abs(neighbourdf['ypos']-neighbourdf['Avail'])
            avg_dist=neighbourdf[['Avail','Distance']].groupby('Avail').mean().reset_index()
            #display(avg_dist)
            best_ypos_distance=avg_dist['Distance'].min()
            best_ypos=avg_dist.loc[avg_dist['Distance']==best_ypos_distance,'Avail']
            best_ypos=best_ypos.iloc[0]#.item()
           
            # If current position is there and is better, keep that as the "best"
            if current_ypos!=None:
                #best_ypos_distance=distance_positions[best_ypos]
                current_ypos_distance=Avg_distance_to_neighbours(node,current_ypos,nodeDf)
                if current_ypos_distance<=best_ypos_distance:
                    best_ypos=current_ypos
            
        else:
            best_ypos=posdf.loc[0,'Avail'].item() # Pick the first available
        

        
        if n%print_update_every==0:
            bsttime_elapsed = (time.time() - bsttime2)
            if showtiming==True:
                print(f'To calc best positions as a vector tooks: {bsttime_elapsed}')
        
        


        # Assign it to the best position
        nodeDf.loc[nodeDf['name']==node,'ypos']=best_ypos # Set it to take the best value


        # Remove available position from the list of available positions
        available_positions=[pos for pos in available_positions if pos!=best_ypos] # Remove the ypos from the list
        #print(available_positions)
        posDf=posDf.loc[~((posDf['xpos']==xpos)&(posDf['y_band']==y_band)*(posDf['y_pos_Available']==best_ypos)),] # And update the dataframe holding available y positions

    if showtiming==True:
        end_time = time. time()
        time_elapsed = (end_time - start_time)
        print(f'Calculating ypos positions themselves took: {time_elapsed}')
        start_time = time. time()
    
    return nodeDf
    
    
    
    
    
    
    
    
    









# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vis5_standardise_pos(node_df):  
    
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return

    # Check that node_df contains all the columns it needs to 
    required_columns=['xpos','ypos'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in node_df.columns:
            print(f'Function requires a field called "{col}". Please consider running vis1_graphDescriptiveStats, vis2_validate_dfs and/or vis3_calc_xpos to generate them')
            exit_flag=True
    if exit_flag==True:
        return
    
    
    ##############################################################################################
    # Scaling
    ##############################################################################################
    max_x=node_df['xpos'].max()
    max_y=node_df['ypos'].max() # Identify the maximum positional width
    y_scale=max_y+1
    x_scale=max_x+1 # Scale between 0 and 1
    
    # Re-scale to be between 0 and 1, with margins  
    node_df['xpos_scaled']=((node_df['xpos']+0.5)/x_scale)#*0.3
    node_df['ypos_scaled']=((node_df['ypos'])/y_scale)#*0.3
    
    return node_df






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def subset_graph(G,node_df,edge_df,keep_nodes='All'):
    # For a given G, node_df and edge_df, subset against a list of nodes
    
    
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that edge_df is a DataFrame
    if not isinstance(edge_df, pd.DataFrame):
        print('ERROR: cannot continue- edge_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return
    
    
     # Check that node_df contains all the columns it needs to 
    required_columns=['name'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in node_df.columns:
            print(f'Function requires a field called "{col}" in node_df.')
            exit_flag=True
    if exit_flag==True:
        return
    
     # Check that node_df contains all the columns it needs to 
    required_columns=['From','To'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in edge_df.columns:
            print(f'Function requires a field called "{col}" in edge_df.')
            exit_flag=True
    if exit_flag==True:
        return
    
    
    
   
    # Check that node_df the same nodes as G
    if len(set(G.nodes()))!=len(set(node_df['name'])):
        print('There is a mismatch between the number of nodes in the Graph object G, and the node_df')
        return
    elif len(set(G.nodes()))!=len(set(list(G.nodes())+list(node_df['name']))):
        print('There is a mismatch between the node names in the Graph object G and the node_Df')
        return
    
    # Validate keep nodes list
    if keep_nodes=='All':
        keep_nodes=list(G.nodes)
    
    if not isinstance(keep_nodes, list):
        print('ERROR: keep_nodes needs to be a list type variable, or the word "All"')
        return
    
    if len(keep_nodes)==0:
        print('ERROR: you must pass at least one node in the keep_nodes list')
        return
    
    
    #########################################################################################################################
    # Subset and return data
    #########################################################################################################################

    # Sub graph
#    subG=G.subgraph(keep_nodes) # Returns only a view of the original graph, not an actual new graph object
    subG=G.to_directed() # Copy the Graph object
    subG.remove_nodes_from([n for n in subG if n not in set(keep_nodes)]) # Remove nodes not in the keep_nodes list
    
    # Subset node df
    sub_nodedf=node_df.merge(pd.DataFrame(keep_nodes,columns=['name']),on='name',how='inner')
    
    # Subset edge_df
    sub_edgedf= edge_df.merge(pd.DataFrame(subG.edges,columns=['From','To']),how='inner',on=['From','To'])
    return subG, sub_nodedf, sub_edgedf





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def concat_text_fields(df,fields):
    
  
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print('ERROR: cannot continue- df parameter should be a DataFrame!')
        return
 
    if not isinstance(fields, list):
        print('ERROR: "fields" parameter needs to be a list type variable')
        return
    
     # Check that node_df contains all the columns it needs to 
    required_columns=fields # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in df.columns:
            print(f'ERROR: {col} not found in node_df. Function requires a field called "{col}" in the dataframe, or for that value not to be included in the "fields" parameter.')
            exit_flag=True
    if exit_flag==True:
        return
    
    #########################################################################################################################
    # Concat of string(s)
    #########################################################################################################################
    df['text']=df.apply(lambda x: '<br>'.join([f'{field}: {str(x[field])}' for field in fields]),axis=1)
    return df



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_pyvis(G,node_df,edge_df,graph_title='Network Graph',canvasSize=(1000,1200),pyvis_size_scale=2
               ,use_images=False
               ,pyvis_x_scale=100
               ,pyvis_y_scale=15
               ,show_graph=True
               ,use_physics=False
               ,savePath='Visualisations_HTML'
                ,buttonlist=['physics','layout'] 
              ,new_opts_as_a_string=None
               ,network_kwargs=None
               ,use_x_as_levels=False
               ,rotate_left=False
              ):
    
    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

    # Check that node_df is a DataFrame
    if not isinstance(node_df, pd.DataFrame):
        print('ERROR: cannot continue- node_df parameter should be a DataFrame!')
        return
    
    # Check that edge_df is a DataFrame
    if not isinstance(edge_df, pd.DataFrame):
        print('ERROR: cannot continue- edge_df parameter should be a DataFrame!')
        return
    
    # Check that G is a Directed Graph
    if not isinstance(G, nx.classes.digraph.DiGraph):
        print('ERROR: cannot continue- G must be a networkX directed Graph!')
        return
    
    
     # Check that node_df contains all the columns it needs to 
    required_columns=['name','xpos','ypos','label','size','text','color'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in node_df.columns:
            print(f'ERROR: Function requires a field called "{col} in node_df".')
            exit_flag=True
    if exit_flag==True:
        return
    
    
    # Check that edge_df contains all the columns it needs to 
    required_columns=['From','To','label','size','text','color'] # Lists the columns that node_df needs to have for this function to work
    exit_flag=False # A toggle to say whether to exit the function
    for col in required_columns:
        if col not in edge_df.columns:
            print(f'Function requires a field called "{col}" in edge_df.')
            exit_flag=True
    if exit_flag==True:
        return
    
   
    # Check that node_df the same nodes as G
    if len(set(G.nodes()))!=len(set(node_df['name'])):
        print('There is a mismatch between the number of nodes in the Graph object G, and the node_df')
        return
    elif len(set(G.nodes()))!=len(set(list(G.nodes())+list(node_df['name']))):
        print('There is a mismatch between the node names in the Graph object G and the node_Df')
        return
    
    
    # Check that canvassize is a 2d thing
    if len(canvasSize)!=2:
        print('canvasSize parameter should be a list of tuple of two items corresponding to canvas height and width')
    
    # Check that use_physics is a boolean
    if not isinstance(use_physics, bool):
        print('ERROR: cannot continue- use_physics parameter should be a boolean (True or False)')
        return
    
    # Check that use_images is a boolean
    if not isinstance(use_images, bool):
        print('ERROR: cannot continue- use_physics parameter should be a boolean (True or False)')
        return
    
    # check that image column is populated if needed
    if use_images==True:
        if 'imageURL' not in node_df.columns:
            print(f'Function requires a field called imageURL in node_df, holding URLs of where to get the images to be used in the layout.')
            return
    
    #########################################################################################################################
    # deep copies to avoid editing the underlying structure
    #########################################################################################################################
    this_G=G
    nodeDf=node_df.copy(deep=True)
    edgeDf=edge_df.copy(deep=False)
    
    #########################################################################################################################
    # It is arranged left-to-right, but sometimes we want to go top to bottom. This flips the view
    #########################################################################################################################
    if rotate_left==True:
        nodeDf['xposbackup']=nodeDf['xpos']
        nodeDf['xpos']=nodeDf['ypos']
        nodeDf['ypos']=nodeDf['xposbackup']

    
    #########################################################################################################################
    # update fields
    #########################################################################################################################
    nodeDf['size']=pyvis_size_scale*nodeDf['size'] # resize with scaling factor, easier than fiddling with node sizes directly
    nodeDf['xpos_scaled']=pyvis_x_scale*nodeDf['xpos'] # resize with scaling factor, easier than fiddling with node sizes directly
    nodeDf['ypos_scaled']=-pyvis_y_scale*nodeDf['ypos'] # resize with scaling factor, and make negative for pyvis
    

    #########################################################################################################################
    # Validation of inputs
    #########################################################################################################################

        # Create a pyvis Network object
    if network_kwargs:
        nt=Network(f"{canvasSize[0]}px",f"{canvasSize[1]}px",directed=True,**network_kwargs)#,notebook)
    else:
        nt=Network(f"{canvasSize[0]}px",f"{canvasSize[1]}px",directed=True)#,notebook)
    
    # Add nodes from the node dataframe

    nt.add_nodes(list(nodeDf['name']),\
                 label=list(nodeDf['label']), # Label to display
                 size=list(nodeDf['size']), # Node size
                 title=list(nodeDf['text']), # The hover text
             color=[i if use_images==False else 'black' for i in list(nodeDf['color'])], # Override to black if absent
             x=list(nodeDf['xpos_scaled']),
             y=list(nodeDf['ypos_scaled']))

        
    # Add a hidden dummy node to centre the graph on
    if use_physics==False:
        nt.add_node(n_id='Dummy',label=' ',size=0,title=None,color=(0,0,0,0),x=canvasSize[0]/2,y=canvasSize[1]/2)

    # Bulk add of edges doesn't support parameters so add one by one
    edgeinfo=edgeDf.to_dict('records')
    for edge in edgeinfo:
        nt.add_edge(edge['From'],edge['To'],                    
               label=edge['label'],
               value=edge['size'],
            title=edge['text'],
               color=edge['color'])
    
    # Turn off the physics so they use the calculated positions, and decide whether to use images
    img_url_list=list(nodeDf['imageURL'])
    img_sizes= list(nodeDf['size'])
    shapes=list(nodeDf['shape'])
    if use_x_as_levels==True:
        levels=list(nodeDf['xpos'])
        dummy_level=max(levels)/2

    for num,n in enumerate(nt.nodes):
        
        if n['id']!='Dummy': 
            n.update({'physics': use_physics}) # Stop pyvis' auto-algorithms overriding
            
            if use_images==True:
                #n.update({'shape':'image'})
                #n.update({'image': img_url_list[num]})
                n.update({'shape':'image','image': img_url_list[num],'size':img_sizes[num]})
            else:
                n.update({'shape':shapes[num]})
            
            
            if use_x_as_levels==True:
                n.update({'level':levels[num]})
        
        else:
            n.update({'physics': True}) # Dummy node always has physics on to centre the graph
            if use_x_as_levels==True:
                n.update({'level':dummy_level})
        
            
            

                        

    nt.set_edge_smooth('continuous')
    nt.show_buttons(buttonlist)
    
    
    if new_opts_as_a_string:
        # The "set_options" feature of pyvis is incomplete because it requires all options to be set with the buttons. If not, you pass only a partial dict and it doesn't work.
        # The below code rectifies that by essentially left joining the changes onto the original, full, JSON options object
        try:
            options_as_a_string=nt.get_network_data()[-1] # Retrieve current as a JSON string 
            print('Got current options string')
            options_as_a_dict=json.loads(options_as_a_string) # Turn them into a dict
            print('Turned it into a dict')
            new_opts_as_a_dict=json.loads(new_opts_as_a_string) #Turn the string of new options into a dict too... these can be pasted from the "generate options" features in pyvis
            print('Turned new options into a dict too')
            options_as_a_dict=update_nested_dict(options_as_a_dict,new_opts_as_a_dict) # Call function from top of module to update nested dict objects
            print('Updated my options')
            options_as_a_string=json.dumps(options_as_a_dict) #Turn it back into a JSON string
            print('Turned it back into a JSON string')
            nt.set_options(options_as_a_string) # And update
            print('And updated the visualisation successfully')            
        except:
            print('ERROR: Unable to update options')
    
    fullSavePath=savePath+f'/{graph_title}-PyVis.html'.replace(':','-').replace('|','-')

    nt.save_graph(fullSavePath)
    if show_graph==True:
        nt.show(fullSavePath)

         
    return nt
    
    
def get_lineage_of_node_list(node_list,G,lineage_type='ancestor'):
    keep_nodes=[]
    for i in node_list:
        keep_nodes.append(i)
        if lineage_type=='ancestor':
            full_list=nx.ancestors(G,i)
        elif lineage_type=='dependent':
            full_list=nx.descendants(G,i)
        for j in full_list:
            if j not in keep_nodes:
                keep_nodes.append(j)
    return keep_nodes

########################################################################################################################################################################
########################################################################################################################################################################
 # DEPRECATED !!!!
########################################################################################################################################################################
########################################################################################################################################################################

def Avg_distance_to_neighbours(node_name,node_pos,node_df):
    
    # Create a df just mapping a node to it's neighbours- one neighbour per row
    neighbourdf=node_df.loc[(node_df['name']==node_name),['name','all_neighbours']].explode('all_neighbours')
    # Get neighbour positions, if they have any
    neighbourdf=pd.merge(neighbourdf,node_df.loc[node_df['ypos'].notna(),['name','ypos']], left_on='all_neighbours',right_on='name')
    # Distance
    neighbourdf['Distance']=neighbourdf['ypos'].apply(lambda x: np.abs(x-node_pos))
    
    if len(neighbourdf['Distance'])==0:
        return 0.0
    else:
        return neighbourdf['Distance'].mean()

    
    