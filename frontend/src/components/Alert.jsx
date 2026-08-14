export default function Alert({ variant = 'info', children }) {
  return (
    <div className={`oncora-alert ${variant}`}>
      <p>{children}</p>
    </div>
  );
}
