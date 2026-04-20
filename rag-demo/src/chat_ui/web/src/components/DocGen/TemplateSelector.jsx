import { useState, useEffect } from 'react';
import { getTemplates } from '../../api/docgen';

export default function TemplateSelector({ lang, onSelect }) {
  const [templates, setTemplates] = useState([]);

  useEffect(() => {
    getTemplates().then(setTemplates).catch(() => {});
  }, []);

  return (
    <div className="template-list">
      {templates.map(tpl => (
        <div key={tpl.template_id} className="template-card" onClick={() => onSelect(tpl.template_id)}>
          <h3>{lang === 'vi' ? tpl.display_name : tpl.display_name_en}</h3>
          <p>{lang === 'vi' ? tpl.description : tpl.description_en}</p>
        </div>
      ))}
    </div>
  );
}
