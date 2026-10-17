export default function rehypeHeadings() {
  return (tree) => {
    visit(tree);
  };
}

function visit(node) {
  if (!node.children) return;

  for (const child of node.children) {
    if (child.type === "element" && ["h2", "h3"].includes(child.tagName)) {
      transformHeading(child);
    }

    visit(child);
  }
}

function transformHeading(node) {
  const text = getText(node);

  const match = text.match(/^(\d+(?:\.\d+)?)\.?\s+(.*)$/);

  if (!match) return;

  const [, number, label] = match;

  node.children = [
  {
    type: "element",
    tagName: "span",
    properties: {
      className: [
        node.tagName === "h2"
          ? "section-number"
          : "subsection-number",
      ],
    },
    children: [
      {
        type: "text",
        value: number.padStart(node.tagName === "h2" ? 2 : 1, "0"),
      },
    ],
  },
  {
    type: "element",
    tagName: "span",
    properties: {},
    children: [{ type: "text", value: label }],
  },
  {
    type: "element",
    tagName: "a",
    properties: {
      className: ["heading-anchor"],
      href: `#${node.properties?.id ?? ""}`,
      ariaLabel: `Link to ${label}`,
    },
    children: [{ type: "text", value: "#" }],
  },
 ];
}

function getText(node) {
  if (node.type === "text") return node.value;
  return (node.children ?? []).map(getText).join("");
}