import React, { useState } from 'react';
import { Send, BarChart3, Brain, TrendingUp, Loader2 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [review, setReview] = useState('');
  const [batchReviews, setBatchReviews] = useState('');
  const [result, setResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('single');
  
  const [sentimentBarImage, setSentimentBarImage] = useState('');
  const [emotionRadarImage, setEmotionRadarImage] = useState('');
  const [modelComparisonImage, setModelComparisonImage] = useState('');
  const [wordCloudImage, setWordCloudImage] = useState('');
  const [sentimentDistImage, setSentimentDistImage] = useState('');

  const analyzeSingleReview = async () => {
    if (!review.trim()) return;
    
    setLoading(true);
    setResult(null);
    setSentimentBarImage('');
    setModelComparisonImage('');
    
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review })
      });
      const data = await response.json();
      setResult(data);
      
      // Fetch visualizations
      const barResponse = await fetch(`${API_URL}/visualize/sentiment-bar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review })
      });
      const barData = await barResponse.json();
      setSentimentBarImage(barData.image);
      
      const compResponse = await fetch(`${API_URL}/visualize/model-comparison`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: review })
      });
      const compData = await compResponse.json();
      setModelComparisonImage(compData.image);
      
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze review. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatchReviews = async () => {
    const reviews = batchReviews.split('\n').filter(r => r.trim());
    if (reviews.length === 0) return;
    
    setLoading(true);
    setBatchResult(null);
    setEmotionRadarImage('');
    setWordCloudImage('');
    setSentimentDistImage('');
    
    try {
      const response = await fetch(`${API_URL}/batch-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviews })
      });
      const data = await response.json();
      setBatchResult(data);
      
      // Fetch batch visualizations
      const radarResponse = await fetch(`${API_URL}/visualize/emotion-radar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviews })
      });
      const radarData = await radarResponse.json();
      setEmotionRadarImage(radarData.image);
      
      const wcResponse = await fetch(`${API_URL}/visualize/wordcloud`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviews })
      });
      const wcData = await wcResponse.json();
      setWordCloudImage(wcData.image);
      
      const distResponse = await fetch(`${API_URL}/visualize/sentiment-distribution`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviews })
      });
      const distData = await distResponse.json();
      setSentimentDistImage(distData.image);
      
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze reviews. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    const colors = {
      'Highly Positive': 'text-green-600 bg-green-100',
      'Positive': 'text-lime-600 bg-lime-100',
      'Neutral': 'text-yellow-600 bg-yellow-100',
      'Negative': 'text-orange-600 bg-orange-100',
      'Strongly Negative': 'text-red-600 bg-red-100'
    };
    return colors[sentiment] || 'text-gray-600 bg-gray-100';
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      'Appreciation': 'bg-blue-500',
      'Satisfaction': 'bg-green-500',
      'Excitement': 'bg-purple-500',
      'Disappointment': 'bg-orange-500',
      'Frustration': 'bg-red-500',
      'Anger': 'bg-red-700'
    };
    return colors[emotion] || 'bg-gray-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8 text-indigo-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">IIIT CourseReview Analyzer</h1>
              <p className="text-sm text-gray-600">Advanced sentiment & emotion analysis powered by ML</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Tabs */}
        <div className="flex space-x-2 mb-6">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'single'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            Single Review
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'batch'
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            Batch Analysis
          </button>
        </div>

        {/* Single Review Tab */}
        {activeTab === 'single' && (
          <div className="space-y-6">
            {/* Input Card */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter Course Review
              </label>
              <textarea
                value={review}
                onChange={(e) => setReview(e.target.value)}
                placeholder="e.g., Excellent course! Prof. Sharma explains algorithms with great clarity..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
                rows={4}
              />
              <button
                onClick={analyzeSingleReview}
                disabled={loading || !review.trim()}
                className="mt-4 w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Analyze Review</span>
                  </>
                )}
              </button>
            </div>

            {/* Results */}
            {result && (
              <div className="space-y-6">
                {/* Sentiment Bar */}
                {sentimentBarImage && (
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                      <TrendingUp className="w-6 h-6 mr-2 text-indigo-600" />
                      Sentiment Analysis
                    </h2>
                    <img src={sentimentBarImage} alt="Sentiment" className="w-full" />
                  </div>
                )}

                {/* Emotions */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Detected Emotions</h2>
                  <div className="flex flex-wrap gap-2">
                    {result.emotions.length > 0 ? (
                      result.emotions.map((emotion, idx) => (
                        <span
                          key={idx}
                          className={`px-4 py-2 rounded-full text-white font-medium ${getEmotionColor(emotion)}`}
                        >
                          {emotion}
                        </span>
                      ))
                    ) : (
                      <span className="text-gray-500 italic">No emotions detected</span>
                    )}
                  </div>
                </div>

                {/* Model Comparison */}
                {modelComparisonImage && (
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                      <BarChart3 className="w-6 h-6 mr-2 text-indigo-600" />
                      Model Comparison
                    </h2>
                    <img src={modelComparisonImage} alt="Model Comparison" className="w-full" />
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-700">
                        <strong>Model Agreement:</strong> {result.model_comparison.agreement.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}

                {/* Top Features */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Top Predictive Features</h2>
                  <div className="space-y-2">
                    {Object.entries(result.feature_importance)
                      .slice(0, 10)
                      .map(([feature, importance], idx) => (
                        <div key={idx} className="flex items-center">
                          <span className="w-32 text-sm text-gray-700 truncate">{feature}</span>
                          <div className="flex-1 mx-4 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-indigo-600 h-2 rounded-full"
                              style={{ width: `${importance * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-600">{(importance * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Batch Analysis Tab */}
        {activeTab === 'batch' && (
          <div className="space-y-6">
            {/* Input Card */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter Multiple Reviews (one per line)
              </label>
              <textarea
                value={batchReviews}
                onChange={(e) => setBatchReviews(e.target.value)}
                placeholder="Excellent course! Prof. Sharma explains concepts clearly.&#10;Disappointing experience. The lectures were hard to follow.&#10;Average course, nothing special."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none font-mono text-sm"
                rows={8}
              />
              <button
                onClick={analyzeBatchReviews}
                disabled={loading || !batchReviews.trim()}
                className="mt-4 w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    <span>Analyze Batch</span>
                  </>
                )}
              </button>
            </div>

            {/* Batch Results */}
            {batchResult && (
              <div className="space-y-6">
                {/* Summary Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <p className="text-sm text-gray-600 mb-1">Total Reviews</p>
                    <p className="text-3xl font-bold text-indigo-600">{batchResult.summary.total_reviews}</p>
                  </div>
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <p className="text-sm text-gray-600 mb-1">Avg Confidence</p>
                    <p className="text-3xl font-bold text-green-600">
                      {(batchResult.summary.avg_confidence * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <p className="text-sm text-gray-600 mb-1">Model Agreement</p>
                    <p className="text-3xl font-bold text-purple-600">
                      {batchResult.summary.avg_model_agreement.toFixed(0)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <p className="text-sm text-gray-600 mb-1">Unique Emotions</p>
                    <p className="text-3xl font-bold text-orange-600">
                      {Object.keys(batchResult.summary.emotion_distribution).length}
                    </p>
                  </div>
                </div>

                {/* Visualizations */}
                {sentimentDistImage && (
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Sentiment Distribution</h2>
                    <img src={sentimentDistImage} alt="Sentiment Distribution" className="w-full" />
                  </div>
                )}

                {emotionRadarImage && (
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Emotion Distribution</h2>
                    <img src={emotionRadarImage} alt="Emotion Radar" className="w-full" />
                  </div>
                )}

                {wordCloudImage && (
                  <div className="bg-white rounded-xl shadow-md p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Word Cloud</h2>
                    <img src={wordCloudImage} alt="Word Cloud" className="w-full" />
                  </div>
                )}

                {/* Individual Results */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Individual Results</h2>
                  <div className="space-y-3">
                    {batchResult.results.map((r, idx) => (
                      <div key={idx} className="p-4 bg-gray-50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-600">Review {idx + 1}</span>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(r.sentiment.ensemble)}`}>
                            {r.sentiment.ensemble}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {r.emotions.map((emo, i) => (
                            <span key={i} className={`px-2 py-1 rounded-full text-xs text-white ${getEmotionColor(emo)}`}>
                              {emo}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
