export default function rehypeStripPreamble() {
  return (tree) => {
    const firstSection = tree.children.findIndex(
      (node) =>
        node.type === "element" &&
        node.tagName === "h2"
    );

    if (firstSection !== -1) {
      tree.children = tree.children.slice(firstSection);
    }
  };
}