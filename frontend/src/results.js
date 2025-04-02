import React from "react";

function Results({ data }) {
  return (
    <div>
      <h3>Results</h3>
      <p><strong>File:</strong> {data.filename}</p>
      <p><strong>Predicted Genre:</strong> {data.predicted_genre}</p>
    </div>
  );
}

export default Results;
