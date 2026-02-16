import { Routes, Route, useNavigate } from "react-router-dom";
import { useState } from "react";
import axios from "axios";

/* ================= MAIN ROUTER ================= */

export default function App() {
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);

  return (
    <Routes>
      <Route
        path="/"
        element={
          <UploadPage
            setResults={setResults}
            setSummary={setSummary}
          />
        }
      />
      <Route
        path="/dashboard"
        element={
          <DashboardPage
            results={results}
            setResults={setResults}
            summary={summary}
            setSummary={setSummary}
          />
        }
      />
    </Routes>
  );
}

/* ================= UPLOAD PAGE ================= */

function UploadPage({ setResults, setSummary }) {
  const navigate = useNavigate();
  const [answerSheet, setAnswerSheet] = useState(null);
  const [answerKey, setAnswerKey] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!answerSheet) {
      alert("Please upload an answer sheet.");
      return;
    }

    const formData = new FormData();
    formData.append("answer_sheet", answerSheet);
    if (answerKey) formData.append("answer_key", answerKey);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:5000/process",
        formData
      );

      const enriched = response.data.results.map(r => ({
        ...r,
        isEditing: false,
        manuallyEdited: false
      }));

      setResults(enriched);
      setSummary({
        total_questions: response.data.total_questions,
        total_correct: response.data.total_correct,
        accuracy: response.data.accuracy
      });

      navigate("/dashboard");

    } catch (error) {
      console.error(error);
      alert("Processing failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-blue-100 to-indigo-100 p-8">
      <div className="max-w-5xl mx-auto bg-white shadow-2xl rounded-3xl p-10">

        <h1 className="text-4xl font-bold text-center mb-10">
          Automated Answer Evaluation System
        </h1>

        <div className="grid md:grid-cols-2 gap-8 mb-8">

          <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-200">
            <label className="block text-lg font-semibold text-indigo-700 mb-4">
              Upload Answer Sheet
            </label>

            <label className="cursor-pointer inline-block bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-xl shadow">
              Choose File
              <input
                type="file"
                onChange={(e) => setAnswerSheet(e.target.files[0])}
                className="hidden"
              />
            </label>

            {answerSheet && (
              <div className="mt-4 bg-white px-4 py-2 rounded-lg border border-indigo-300 text-indigo-700 font-medium">
                {answerSheet.name}
              </div>
            )}
          </div>

          <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-200">
            <label className="block text-lg font-semibold text-indigo-700 mb-4">
              Upload Answer Key (PDF)
            </label>

            <label className="cursor-pointer inline-block bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-xl shadow">
              Choose File
              <input
                type="file"
                onChange={(e) => setAnswerKey(e.target.files[0])}
                className="hidden"
              />
            </label>

            {answerKey && (
              <div className="mt-4 bg-white px-4 py-2 rounded-lg border border-indigo-300 text-indigo-700 font-medium">
                {answerKey.name}
              </div>
            )}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-4 rounded-2xl font-semibold shadow-md"
        >
          {loading ? "Processing..." : "Run Evaluation"}
        </button>

      </div>
    </div>
  );
}

/* ================= DASHBOARD PAGE ================= */

function DashboardPage({ results, setResults, summary, setSummary }) {
  const navigate = useNavigate();

  const [filterText, setFilterText] = useState("");
  const [sortOrder, setSortOrder] = useState("asc");
  const [remarkFilter, setRemarkFilter] = useState("all");

  const recalculateSummary = (updated) => {
    let totalQuestions = 0;
    let totalCorrect = 0;

    updated.forEach(item => {
      if (item.correct_answer !== "—") {
        totalQuestions++;
        if (item.remark === "Correct") totalCorrect++;
      }
    });

    const accuracy =
      totalQuestions > 0
        ? ((totalCorrect / totalQuestions) * 100).toFixed(2)
        : 0;

    setSummary({
      total_questions: totalQuestions,
      total_correct: totalCorrect,
      accuracy
    });
  };

  const handleEdit = (index, value) => {
    const updated = [...results];
    updated[index].detected_answer = value;
    updated[index].manuallyEdited = true;

    updated[index].remark =
      value === updated[index].correct_answer
        ? "Correct"
        : "Wrong";

    setResults(updated);
    recalculateSummary(updated);
  };

  // ---------------- FILTER LOGIC ----------------

  let displayed = results.filter(item =>
    item.question.toString().includes(filterText)
  );

  if (remarkFilter === "manual") {
    displayed = displayed.filter(item => item.manuallyEdited);
  } else if (remarkFilter !== "all") {
    displayed = displayed.filter(
      item => item.remark === remarkFilter
    );
  }

  displayed = displayed.sort((a, b) =>
    sortOrder === "asc"
      ? a.question - b.question
      : b.question - a.question
  );

  const exportFinalReport = async () => {
    await axios.post(
      "http://127.0.0.1:5000/save_corrections",
      { results }
    );

    window.open(
      "http://127.0.0.1:5000/download_results",
      "_blank"
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-blue-100 to-indigo-100 p-8">

      <div className="max-w-6xl mx-auto bg-white shadow-2xl rounded-3xl p-10">

        <div className="flex justify-between items-center mb-10">
          <h1 className="text-3xl font-bold">
            Evaluation Dashboard
          </h1>

          <div className="flex gap-4">
            <button
              onClick={exportFinalReport}
              className="bg-indigo-700 text-white px-6 py-2 rounded-xl"
            >
              Download Final Report
            </button>

            <button
              onClick={() => navigate("/")}
              className="bg-gray-700 text-white px-6 py-2 rounded-xl"
            >
              Back
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-200 mb-8 grid md:grid-cols-3 gap-6">
          <input
            placeholder="Search Question No"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="border px-4 py-2 rounded-xl"
          />

          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className="border px-4 py-2 rounded-xl"
          >
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>

          <select
            value={remarkFilter}
            onChange={(e) => setRemarkFilter(e.target.value)}
            className="border px-4 py-2 rounded-xl"
          >
            <option value="all">All</option>
            <option value="Correct">Correct</option>
            <option value="Wrong">Wrong</option>
            <option value="Unable to read">Unable to read</option>
            <option value="manual">Manually Corrected</option>
          </select>
        </div>

        {/* Summary */}
        {summary && (
          <div className="grid md:grid-cols-3 gap-6 mb-10 text-center">
            <div className="bg-gray-50 p-6 rounded-xl shadow">
              <p>Total Questions</p>
              <p className="text-3xl font-bold">
                {summary.total_questions}
              </p>
            </div>

            <div className="bg-gray-50 p-6 rounded-xl shadow">
              <p>Total Correct</p>
              <p className="text-3xl font-bold text-green-600">
                {summary.total_correct}
              </p>
            </div>

            <div className="bg-gray-50 p-6 rounded-xl shadow">
              <p>Accuracy</p>
              <p className="text-3xl font-bold text-indigo-600">
                {summary.accuracy}%
              </p>
            </div>
          </div>
        )}

        {/* Results (NO PAGINATION) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {displayed.map((item, index) => (
            <div
              key={index}
              className="bg-gray-50 p-6 rounded-2xl shadow-md hover:shadow-xl transition-all"
            >
              <h2 className="text-xl font-bold mb-4">
                Question {item.question}
              </h2>

              <div className="flex gap-6 items-start">
                <img
                  src={`http://127.0.0.1:5000${item.image_url}`}
                  alt="answer"
                  className="border rounded-lg w-24"
                />

                <div className="space-y-3">

                  <div>
                    <strong>Detected Answer:</strong>{" "}

                    {item.isEditing ? (
                      <>
                        <input
                          value={item.tempValue ?? item.detected_answer ?? ""}
                          onChange={(e) => {
                            const updated = [...results];
                            const idx = results.indexOf(item);
                            updated[idx].tempValue = e.target.value;
                            setResults(updated);
                          }}
                          className="border rounded px-2 py-1 ml-2"
                        />

                        <button
                          onClick={() => {
                            const updated = [...results];
                            const idx = results.indexOf(item);

                            const newValue = updated[idx].tempValue ?? "";

                            updated[idx].detected_answer = newValue;
                            updated[idx].manuallyEdited = true;
                            updated[idx].isEditing = false;
                            updated[idx].tempValue = undefined;

                            updated[idx].remark =
                              newValue === updated[idx].correct_answer
                                ? "Correct"
                                : "Wrong";

                            setResults(updated);
                            recalculateSummary(updated);
                          }}
                          className="ml-2 bg-indigo-600 text-white px-3 py-1 rounded"
                        >
                          Save
                        </button>
                      </>
                    ) : (
                      <>
                        <span className="ml-2 font-semibold">
                          {item.detected_answer || "-"}
                        </span>

                        <button
                          onClick={() => {
                            const updated = [...results];
                            const idx = results.indexOf(item);
                            updated[idx].isEditing = true;
                            updated[idx].tempValue = updated[idx].detected_answer;
                            setResults(updated);
                          }}
                          className="ml-3 bg-gray-800 text-white px-3 py-1 rounded"
                        >
                          Edit
                        </button>
                      </>
                    )}
                  </div>


                  <div>
                    <strong>Correct Answer:</strong>{" "}
                    {item.correct_answer}
                  </div>

                  <div>
                    <strong>Remark:</strong>{" "}
                    <span
                      className={`font-bold ${
                        item.remark === "Correct"
                          ? "text-green-600"
                          : item.remark === "Wrong"
                          ? "text-red-600"
                          : "text-orange-500"
                      }`}
                    >
                      {item.remark}
                    </span>
                  </div>

                  <div>
                    <strong>Confidence:</strong>{" "}
                    {item.manuallyEdited
                      ? "Manually Corrected"
                      : `${item.confidence}%`}
                  </div>

                </div>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
