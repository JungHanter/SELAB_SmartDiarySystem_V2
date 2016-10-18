def demo():
    """
    A demonstration showing how Trees and Trees can be
    used.  This demonstration creates a Tree, and loads a
    Tree from the Treebank corpus,
    and shows the results of calling several of their methods.
    """

    from nltk import Tree, ProbabilisticTree

    # Demonstrate tree parsing.
    s = '(S (NP (DT the) (NN sand) (NN ocean) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
    t = Tree.fromstring(s)
    print("Convert bracketed string into tree:")
    print(t)
    print(t.__repr__())

    print("Display tree properties:")
    print(t.label())         # tree's constituent type
    print(t[0])             # tree's first child
    print(t[1])             # tree's second child
    # print(t.height())
    print(t.leaves())
    print(t[1])
    print(t[1,1])
    print(t[1,1,0])

    # Demonstrate tree modification.
    the_cat = t[0]
    the_cat.insert(1, Tree.fromstring('(JJ big)'))
    print("Tree modification:")
    print(t)
    t[1,1,1] = Tree.fromstring('(NN cake)')
    print(t)
    print()

    # Tree transforms
    print("Collapse unary:")
    t.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="a")
    print(t)
    print("Chomsky normal form:")
    t.chomsky_normal_form()
    print(t)
    print()

    # Demonstrate probabilistic trees.
    pt = ProbabilisticTree('x', ['y', 'z'], prob=0.5)
    print("Probabilistic Tree:")
    print(pt)
    print()

    # Demonstrate parsing of treebank output format.
    t = Tree.fromstring(t.pformat())
    print("Convert tree to bracketed string and back again:")
    print(t)
    print()

    # Demonstrate LaTeX output
    print("LaTeX output:")
    print(t.pformat_latex_qtree())
    print()

    # Demonstrate Productions
    print("Production output:")
    print(t.productions())
    print()

    # Demonstrate tree nodes containing objects other than strings
    t.set_label(('test', 3))
    print(t)

if __name__ == '__main__':
    demo()