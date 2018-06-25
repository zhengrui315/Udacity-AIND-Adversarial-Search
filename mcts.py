## Monte Carlo Tree Search

import random, math, copy


class MCTS_Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = MCTS_Node(child_state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        return len(self.children_actions) == len(self.state.actions())


FACTOR = 1.0
iter_limit = 100

# def mcts(state):
#     root = MCTS_Node(state)
#     for _ in range(iter_limit):
#         child = tree_policy(root)
#         reward = default_policy(child.state)
#         backup(child,reward)
#
#     idx = root.children.index(best_child(root))
#     return root.children_actions[idx]

def tree_policy(node):
    """
    Select a leaf node.
    If not fully explored, return an unexplored child node.
    Otherwise, return the child node with the best score.

    :param node:
    :return: node
    """
    while not node.state.terminal_test():
        if not node.fully_explored():
            return expand(node)
        node = best_child(node)
    return node


def expand(node):
    tried_actions = node.children_actions
    legal_actions = node.state.actions()
    for action in legal_actions:
        if action not in tried_actions:
            new_state = node.state.result(action)
            node.add_child(new_state, action)
            return node.children[-1]


def best_child(node):
    """
    Find the child node with the best score.

    :param node:
    :return: node;
    """
    best_score = float("-inf")
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
        score = exploit + FACTOR * explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score
    # if len(best_children) == 0:
    #     print("WARNING - RuiZheng, there is no best child")
    #     return None
    return random.choice(best_children)


def default_policy(state):
    """
    Randomly search the descendant of the state, and return the reward

    :param state:
    :return: int
    """
    init_state = copy.deepcopy(state)
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)

    # let the reward be 1 for the winner, -1 for the loser
    # if the init_state.player() wins, it means the action that leads to
    # init_state should be discouraged, so reward = -1.
    return -1 if state._has_liberties(init_state.player()) else 1


def backup(node, reward):
    """
    Backpropagation
    Use the result to update information in the nodes on the path.

    :param node:
    :param reward: int
    :return:
    """
    while node != None:
        node.update(reward)
        node = node.parent
        reward *= -1


