import { useEffect, useRef, useState } from 'react';

export default function MriUploader({ file, onChange, label = 'Upload MRI Image' }) {
  const inputRef = useRef(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div
      className="oncora-file-drop"
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={(e) => onChange(e.target.files?.[0] || null)}
      />
      <div className="oncora-file-drop-label">
        {file ? file.name : `${label} - JPG, JPEG or PNG`}
      </div>
      {previewUrl && <img src={previewUrl} alt="Uploaded MRI preview" />}
    </div>
  );
}
