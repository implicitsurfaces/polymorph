import type {  NumNode } from "../../num-tree";
import type { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

export class DotEvalKernel implements NumEvalKernel<number> {
  private body: string[] = [];
  private currentId = 1;

  constructor(
    public readonly variablesValues: Map<string, number> = new Map(),
  ) {}

  private newId() {
    return this.currentId++;
  }

  private addNode(
    id: number,
    label: string,
    status: "info" | "warn" | "error" = "info",
  ) {
    const color =
      status === "warn" ? "yellow" : status === "error" ? "red" : "lightblue";
    this.body.push(`    node${id} [label="${label}" fillcolor=${color}];`);
  }

  private addEdge(from: number, to: number) {
    this.body.push(`    node${from} -> node${to};`);
  }

  private formatLabel(operation: string, value: number) {
    let label = operation.toLowerCase();

    if (value || value === 0 || Number.isNaN(value)) {
      label += ` (${Number.isNaN(value) ? "NaN" : value?.toFixed(4)})`;
    }

    return label;
  }

  private getStatus(value: number) {
    return Number.isNaN(value)
      ? "error"
      : value !== 0 && Math.abs(value) < 1e-6
        ? "warn"
        : "info";
  }

  value(value: number) {
    return value;
  }

  literal(value: number) {
    const id = this.newId();
    const label = value.toFixed(5);
    this.addNode(id, label);
    return id;
  }

  variable(name: string, node: NumNode & { evalsTo: number }) {
    const id = this.newId();
    const label = this.formatLabel(name, node.evalsTo);
    this.addNode(id, label, this.getStatus(node.evalsTo));
    return id;
  }

  unaryOp(
    operation: UnaryOperation,
    operand: number,
    node: NumNode & { evalsTo: number },
  ) {
    const id = this.newId();
    const label = this.formatLabel(operation, node.evalsTo);
    this.addNode(id, label, this.getStatus(node.evalsTo));
    this.addEdge(id, operand);
    return id;
  }

  binaryOp(
    operation: BinaryOperation,
    left: number,
    right: number,
    node: NumNode & { evalsTo: number },
  ) {
    const id = this.newId();
    const label = this.formatLabel(operation, node.evalsTo);
    this.addNode(id, label, this.getStatus(node.evalsTo));
    this.addEdge(id, left);
    this.addEdge(id, right);
    return id;
  }

  getDot() {
    const lines = ["digraph ExpressionTree {"];
    lines.push("    node [shape=circle, style=filled, fillcolor=lightblue];");
    lines.push(...this.body);
    lines.push("}");
    return lines.join("\n");
  }
}
