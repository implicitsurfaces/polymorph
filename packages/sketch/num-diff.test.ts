digraph ExpressionTree {
    node [shape=circle, style=filled, fillcolor=lightblue];
    node0 [label="ADD"];
    node1 [label="MUL"];
    node2 [label="d/dx"];
    node3 [label="x"];
    node2 -> node3;
    node4 [label="x"];
    node1 -> node2;
    node1 -> node4;
    node5 [label="MUL"];
    node6 [label="x"];
    node7 [label="d/dx"];
    node8 [label="x"];
    node7 -> node8;
    node5 -> node6;
    node5 -> node7;
    node0 -> node1;
    node0 -> node5;
}