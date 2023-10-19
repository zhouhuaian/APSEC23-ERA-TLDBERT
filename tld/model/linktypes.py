from typing import Literal, Dict

fine_linktype_map = {
    'Account': 'Impacts',
    
    'Backports': 'Backport', 

    'Blocked': 'Block',
    'Blocker': 'Block',
    'Blocks': 'Block',

    'Bonfire Testing': 'Bonfire Testing', 
    'Bonfire testing': 'Bonfire Testing', 
    'Git Code Review': 'Bonfire Testing', 
    'Testing': 'Bonfire Testing',

    'Causality': 'Cause', 
    'Cause': 'Cause',
    'Caused': 'Cause', 
    'Problem/Incident': 'Cause',

    'Child-Issue': 'Parent-Child', 
    'Parent Feature': 'Parent-Child',
    'Parent/Child': 'Parent-Child',
    'multi-level hierarchy [GANTT]': 'Parent-Child',
    'Parent-Relation': 'Parent-Child',

    'Cloners': 'Clone', 
    'Cloners (old)': 'Clone', 
    'Cloners (migrated)' : 'Clone',
    
    'Covered': 'Cover',

    'Collection': 'Incorporate', 
    'Container': 'Incorporate',
    'Contains(WBSGantt)': 'Incorporate', 
    'Incorporate': 'Incorporate', 
    'Incorporates': 'Incorporate', 
    'Part': 'Incorporate',
    'PartOf': 'Incorporate',
    'Superset': 'Incorporate', 

    'Completes': 'Fix', 
    'Fixes': 'Fix',
    'Resolve': 'Fix',

    'Depend': 'Depend', 
    'Dependency': 'Depend', 
    'Dependent': 'Depend', 
    'Depends': 'Depend', 
    'Gantt Dependency': 'Depend',
    'dependent': 'Depend',

    'Derived': 'Derive',

    'Detail': 'Detail', 

    'Documentation': 'Documented',
    'Documented': 'Documented',
    
    'Duplicate': 'Duplicate',

    'Epic': 'Epic', 
    'Epic-Relation': 'Epic',
    'Initiative': 'Epic',
    
    'Finish-to-Finish link (WBSGantt)': 'finish-finish', 
    'Gantt End to End': 'finish-finish', 
    'Gantt: finish-finish': 'finish-finish',
    'finish-finish [GANTT]': 'finish-finish', 
    
    'Gantt End to Start': 'finish-start', 
    'Gantt: finish-start': 'finish-start',
    'finish-start [GANTT]': 'finish-start',

    'Gantt Start to Start': 'start-start', 
    'Gantt: start-start': 'start-start',
   
    'Gantt: start-finish': 'start-finish', 
    'start-finish [GANTT]': 'start-finish', 
    
    'Follows': 'Follow', 
    'Sequence': 'Follow', 
    
    'Implement': 'Implement', 
    'Implements': 'Implement', 
    
    'Issue split': 'Split',
    'Split': 'Split',
    'Work Breakdown': 'Split',
    
    'Preceded By': 'Precede', 
    
    'Reference': 'Relate',
    'Relate': 'Relate',
    'Related': 'Relate', 
    'Relates': 'Relate',
    'Relationship': 'Relate',
    
    'Regression': 'Breaks', 
    
    'Replacement': 'Replace',
    
    'Required': 'Require', 
    
    'Supercedes': 'Supercede',
    'Supersede': 'Supercede',
    'Supersession': 'Supercede', 
    
    'Subtask': 'Subtask',
    
    'Test': 'Test', 
    'Tested': 'Test',
    
    'Trigger': 'Trigger', 
          
    '1 - Relate': 'Relate',
    '5 - Depend':   'Depend',          
    '3 - Duplicate': 'Duplicate',          
    '4 - Incorporate': 'Incorporate',        
    '2 - Cloned': 'Clone',    
    '6 - Blocks': 'Block',     
    '7 - Git Code Review': 'Bonfire Testing',
    'Verify': 'Verify'
}

# fine_linktype_to_category_map = {
#     'Backport': 'Workflow',
#     'Block': 'Causal',
#     'Bonfire Testing': 'Workflow',
#     'Breaks': 'Causal',
#     'Cause': 'Causal',
#     'Clone': 'Duplicate',
#     'Cover': 'Workflow',
#     'Depend': 'Causal',
#     'Derive': 'Workflow',
#     'Detail': 'Workflow',
#     'Documented': 'Workflow',
#     'Duplicate': 'Duplicate',
#     'Epic': 'Composition',
#     'Fix': 'Workflow',
#     'Follow': 'Causal',
#     'Impacts': 'Causal',
#     'Incorporate': 'Composition',
#     'Parent-Child': 'Composition',
#     'Precede': 'Causal',
#     'Relate': 'General',
#     'Replace': 'Workflow',
#     'Require': 'Causal',
#     'Split': 'Composition',
#     'Subtask': 'Composition',
#     'Supercede': 'Causal',
#     'Test': 'Workflow',
#     'Trigger': 'Workflow',
#     'finish-start': 'Causal',
#     'finish-finish': 'Causal',
#     'start-start': 'Causal',
#     'start-finish': 'Causal',
#     'Verify': 'Workflow'
# }

fine_linktype_to_category_map = {
    'Backport': 'Linked',
    'Block': 'Linked',
    'Bonfire Testing': 'Linked',
    'Breaks': 'Linked',
    'Cause': 'Linked',
    'Clone': 'Linked',
    'Cover': 'Linked',
    'Depend': 'Linked',
    'Derive': 'Linked',
    'Detail': 'Linked',
    'Documented': 'Linked',
    'Duplicate': 'Linked',
    'Epic': 'Linked',
    'Fix': 'Linked',
    'Follow': 'Linked',
    'Impacts': 'Linked',
    'Incorporate': 'Linked',
    'Parent-Child': 'Linked',
    'Precede': 'Linked',
    'Relate': 'Linked',
    'Replace': 'Linked',
    'Require': 'Linked',
    'Split': 'Linked',
    'Subtask': 'Linked',
    'Supercede': 'Linked',
    'Test': 'Linked',
    'Trigger': 'Linked',
    'finish-start': 'Linked',
    'finish-finish': 'Linked',
    'start-start': 'Linked',
    'start-finish': 'Linked',
    'Verify': 'Linked'
}

category_map = {
    input_linktype: fine_linktype_to_category_map[fine_linktype]
    for input_linktype, fine_linktype in fine_linktype_map.items()
    if fine_linktype in fine_linktype_to_category_map
}

non_link_name = 'Non-Link'

Target = Literal['linktype', 'category']

linktype_map: Dict[Target, Dict[str, str]] = {
    'linktype': fine_linktype_map,
    'category': category_map,
}
