import React, { useState } from "react";
import axios from "axios";
import Results from "./Results";

function uploadstuff() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <div>
      <input type="file" accept=".mp3,.wav" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Predict</button>
      {results && <Results data={results} />}
    </div>
  );
}

export default uploadstuff;
