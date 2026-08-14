export default function Expander({ title, children }) {
  return (
    <details className="oncora-expander">
      <summary>{title}</summary>
      <div>{children}</div>
    </details>
  );
}
