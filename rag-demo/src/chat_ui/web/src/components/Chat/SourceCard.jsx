import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

/**
 * Build a Scroll-to-Text-Fragment URL.
 * source_url + #:~:text= + URL-encoded article citation (dieu)
 */
function buildHighlightUrl(src) {
  const baseUrl = src.source_url;
  if (!baseUrl) return '#';

  // Use the dieu (article reference) as the highlight text
  const highlightText = src.dieu;
  if (!highlightText) return baseUrl;

  const encoded = encodeURIComponent(highlightText);
  return `${baseUrl}#:~:text=${encoded}`;
}

export default function SourceCard({ label, sources, type }) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  // RAG: only show top 1 result
  const displaySources = type === 'rag' ? sources.slice(0, 3) : sources;

  return (
    <div className="source-cards">
      <button className="source-toggle" onClick={() => setExpanded(!expanded)}>
        {label} {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {expanded && (
        <div className="source-list">
          {displaySources.map((src, i) => (
            <div key={i} className="source-item">
              {type === 'rag' ? (
                <>
                  <a href={buildHighlightUrl(src)} target="_blank" rel="noopener noreferrer">
                    {src.dieu || src.luat || src.filename || `Nguồn ${i + 1}`}
                  </a>
                  <p>{(src.text || '').slice(0, 150)}{src.text?.length > 150 ? '...' : ''}</p>
                </>
              ) : (
                <>
                  <a href={src.highlight_url || src.url || '#'} target="_blank" rel="noopener noreferrer">
                    {src.title || `Nguồn ${i + 1}`}
                  </a>
                  <p>{(src.snippet || '').slice(0, 150)}{src.snippet?.length > 150 ? '...' : ''}</p>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
