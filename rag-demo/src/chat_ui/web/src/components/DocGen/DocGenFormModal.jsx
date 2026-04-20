import { useState, useEffect } from 'react';
import { getTemplate, generateDocument } from '../../api/docgen';
import { strings } from '../../i18n/strings';
import toast from 'react-hot-toast';

export default function DocGenFormModal({ templateId, lang, onClose, onComplete }) {
  const t = strings[lang];
  const [templateDef, setTemplateDef] = useState(null);
  const [fields, setFields] = useState({});
  const [errors, setErrors] = useState({});
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    getTemplate(templateId).then(data => {
      setTemplateDef(data);
      // Initialize field values
      const init = {};
      data.fields.forEach(f => { init[f.name] = ''; });
      setFields(init);
    }).catch(err => {
      toast.error('Failed to load template');
      onClose();
    });
  }, [templateId]);

  const handleChange = (fieldName, value) => {
    setFields(prev => ({ ...prev, [fieldName]: value }));
    if (errors[fieldName]) {
      setErrors(prev => { const n = { ...prev }; delete n[fieldName]; return n; });
    }
  };

  const validate = () => {
    const errs = {};
    templateDef.fields.forEach(f => {
      if (f.required && !fields[f.name]?.trim()) {
        errs[f.name] = lang === 'vi' ? 'Trường bắt buộc' : 'Required field';
      }
    });
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleGenerate = async () => {
    if (!validate()) return;
    setGenerating(true);
    try {
      const result = await generateDocument(templateId, fields);
      // Download file
      const byteCharacters = atob(result.content_base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: result.mime_type });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast.success(t.docgenSuccess);
      onComplete();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Lỗi tạo đơn');
    } finally {
      setGenerating(false);
    }
  };

  if (!templateDef) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <div className="modal-body" style={{ textAlign: 'center', padding: 40 }}>
            <div className="typing-indicator" style={{ justifyContent: 'center' }}>
              <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Group fields by group
  const groups = [];
  const groupMap = new Map();
  templateDef.fields.forEach(f => {
    const groupName = lang === 'vi' ? f.group : (f.group_en || f.group);
    if (!groupMap.has(groupName)) {
      groupMap.set(groupName, []);
      groups.push(groupName);
    }
    groupMap.get(groupName).push(f);
  });

  const filledRequired = templateDef.fields.filter(f => f.required && fields[f.name]?.trim()).length;
  const totalRequired = templateDef.fields.filter(f => f.required).length;
  const progress = totalRequired > 0 ? Math.round((filledRequired / totalRequired) * 100) : 100;

  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-content">
        <div className="modal-header">
          <div>
            <h2>📋 {lang === 'vi' ? templateDef.display_name : templateDef.display_name_en}</h2>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              {filledRequired}/{totalRequired} {lang === 'vi' ? 'trường bắt buộc' : 'required fields'} ({progress}%)
            </div>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {groups.map(groupName => (
            <div key={groupName}>
              <div className="form-group-title">{groupName}</div>
              {groupMap.get(groupName).map(field => (
                <div key={field.name} className="form-field">
                  <label>
                    {lang === 'vi' ? field.label : field.label_en}
                    {field.required && <span className="required-dot">*</span>}
                  </label>
                  {field.field_type === 'textarea' ? (
                    <textarea
                      className={errors[field.name] ? 'field-error' : ''}
                      placeholder={field.placeholder}
                      value={fields[field.name] || ''}
                      onChange={e => handleChange(field.name, e.target.value)}
                    />
                  ) : (
                    <input
                      type="text"
                      className={errors[field.name] ? 'field-error' : ''}
                      placeholder={field.placeholder}
                      value={fields[field.name] || ''}
                      onChange={e => handleChange(field.name, e.target.value)}
                    />
                  )}
                  {errors[field.name] && <div className="error-text">{errors[field.name]}</div>}
                </div>
              ))}
            </div>
          ))}

          <div className="docgen-disclaimer">{t.docgenDisclaimer}</div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-ghost" onClick={onClose}>{t.cancel}</button>
          <button className="btn btn-primary" onClick={handleGenerate} disabled={generating}>
            {generating ? t.generating : t.generate}
          </button>
        </div>
      </div>
    </div>
  );
}
