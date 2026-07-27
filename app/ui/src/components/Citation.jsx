import React from 'react';

const Citation = ({ source }) => {
  return (
    <div className="citation-card" title={source.content}>
      <div className="citation-header">
        <span className="citation-part">{source.part_number}</span>
        <span className="citation-page">Page {source.page}</span>
      </div>
      <div className="citation-section">{source.section}</div>
      <div className="citation-content">{source.content}</div>
    </div>
  );
};

export default Citation;
