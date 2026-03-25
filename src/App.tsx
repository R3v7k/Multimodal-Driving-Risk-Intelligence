import React, { useState } from 'react';
import { Car, AlertTriangle, Activity, Info, Upload, Zap } from 'lucide-react';

const PROVIDERS = {
  gemini: {
    name: "Gemini",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-3.1-pro-preview"]
  },
  openai: {
    name: "OpenAI",
    models: ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
  },
  claude: {
    name: "Claude",
    models: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
  },
  deepseek: {
    name: "DeepSeek",
    models: ["deepseek-chat", "deepseek-coder"]
  }
};

export default function App() {
  const [speed, setSpeed] = useState(45);
  const [weather, setWeather] = useState(0);
  const [timeOfDay, setTimeOfDay] = useState(14);
  const [alertness, setAlertness] = useState(0.8);
  const [report, setReport] = useState("Sudden braking detected while approaching an intersection.");
  const [provider, setProvider] = useState<keyof typeof PROVIDERS>("gemini");
  const [selectedModel, setSelectedModel] = useState(PROVIDERS.gemini.models[0]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    // Simulate API call to the Python backend
    setTimeout(() => {
      setResult({
        hazard_category: 2,
        hazard_name: "Pedestrian",
        risk_score: 0.85,
        provider_used: `${PROVIDERS[provider].name} (${selectedModel})`,
        reasoning: "The combination of sudden braking and the presence of a pedestrian at the intersection indicates a high risk of collision. Immediate driver intervention or automated emergency braking is recommended."
      });
      setIsAnalyzing(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col md:flex-row">
      {/* Sidebar */}
      <div className="w-full md:w-80 bg-white border-r border-slate-200 p-6 flex flex-col h-screen overflow-y-auto shadow-sm z-10">
        <div className="flex items-center gap-2 mb-8">
          <div className="bg-blue-600 p-2 rounded-lg text-white">
            <Car size={24} />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-800">Risk Intel</h1>
        </div>

        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">Incident Data Input</h2>
        
        <div className="space-y-6">
          {/* Image Upload */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Dashcam Image</label>
            <label htmlFor="image-upload" className="block border-2 border-dashed border-slate-300 rounded-lg p-4 text-center hover:bg-slate-50 transition-colors cursor-pointer">
              <Upload className="mx-auto text-slate-400 mb-2" size={20} />
              <span className="text-sm text-slate-500 truncate block px-2">
                {imageFile ? imageFile.name : "Upload JPG, PNG"}
              </span>
              <input 
                id="image-upload" 
                type="file" 
                accept="image/jpeg, image/png" 
                className="hidden" 
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    setImageFile(e.target.files[0]);
                  }
                }} 
              />
            </label>
          </div>

          {/* Speed */}
          <div>
            <div className="flex justify-between mb-1">
              <label className="text-sm font-medium text-slate-700">Speed (mph)</label>
              <span className="text-sm text-slate-500">{speed}</span>
            </div>
            <input type="range" min="0" max="120" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full accent-blue-600" />
          </div>

          {/* Weather */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Weather Condition</label>
            <select value={weather} onChange={(e) => setWeather(Number(e.target.value))} className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none">
              <option value={0}>Clear</option>
              <option value={1}>Rain</option>
              <option value={2}>Snow</option>
              <option value={3}>Fog</option>
            </select>
          </div>

          {/* Time of Day */}
          <div>
            <div className="flex justify-between mb-1">
              <label className="text-sm font-medium text-slate-700">Time of Day (Hour)</label>
              <span className="text-sm text-slate-500">{timeOfDay}:00</span>
            </div>
            <input type="range" min="0" max="23" value={timeOfDay} onChange={(e) => setTimeOfDay(Number(e.target.value))} className="w-full accent-blue-600" />
          </div>

          {/* Alertness */}
          <div>
            <div className="flex justify-between mb-1">
              <label className="text-sm font-medium text-slate-700">Driver Alertness</label>
              <span className="text-sm text-slate-500">{alertness.toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={alertness} onChange={(e) => setAlertness(Number(e.target.value))} className="w-full accent-blue-600" />
          </div>

          {/* Report Text */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Incident Report</label>
            <textarea 
              value={report} 
              onChange={(e) => setReport(e.target.value)} 
              className="w-full border border-slate-300 rounded-md p-2 text-sm h-24 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-none"
            />
          </div>

          {/* Provider */}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">LLM Provider</label>
            <select 
              value={provider} 
              onChange={(e) => {
                const newProvider = e.target.value as keyof typeof PROVIDERS;
                setProvider(newProvider);
                setSelectedModel(PROVIDERS[newProvider].models[0]);
              }} 
              className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none mb-3"
            >
              {Object.entries(PROVIDERS).map(([key, data]) => (
                <option key={key} value={key}>{data.name}</option>
              ))}
            </select>

            <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)} 
              className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              {PROVIDERS[provider].models.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8 bg-slate-50 h-screen overflow-y-auto">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Dashboard</h2>
              <p className="text-slate-500">Multimodal Driving Risk Assessment</p>
            </div>
            <button 
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium flex items-center gap-2 transition-colors disabled:opacity-70"
            >
              {isAnalyzing ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Zap size={20} />
              )}
              {isAnalyzing ? 'Analyzing...' : 'Analyze Risk'}
            </button>
          </div>

          {!result && !isAnalyzing && (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm">
              <Activity className="mx-auto text-slate-300 mb-4" size={48} />
              <h3 className="text-lg font-medium text-slate-900 mb-2">Ready for Analysis</h3>
              <p className="text-slate-500 max-w-md mx-auto">
                Configure the incident parameters in the sidebar and click "Analyze Risk" to process the multimodal data through the PyTorch and Gemini models.
              </p>
            </div>
          )}

          {isAnalyzing && (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm">
              <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-900">Processing Multimodal Data...</h3>
              <p className="text-slate-500">Running ResNet vision backbone and querying {PROVIDERS[provider].name} ({selectedModel})...</p>
            </div>
          )}

          {result && !isAnalyzing && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Metrics Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="bg-orange-100 text-orange-600 p-2 rounded-lg">
                      <AlertTriangle size={20} />
                    </div>
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Hazard Category</h3>
                  </div>
                  <p className="text-3xl font-bold text-slate-900">{result.hazard_name}</p>
                </div>

                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="bg-red-100 text-red-600 p-2 rounded-lg">
                      <Activity size={20} />
                    </div>
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Risk Score</h3>
                  </div>
                  <div className="flex items-end gap-2">
                    <p className="text-3xl font-bold text-slate-900">{result.risk_score.toFixed(2)}</p>
                    <p className="text-sm text-slate-500 mb-1">/ 1.00</p>
                  </div>
                </div>

                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="bg-purple-100 text-purple-600 p-2 rounded-lg">
                      <Zap size={20} />
                    </div>
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Provider</h3>
                  </div>
                  <p className="text-3xl font-bold text-slate-900 capitalize">{result.provider_used}</p>
                </div>
              </div>

              {/* Reasoning Card */}
              <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
                    <Info size={20} />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900">AI Reasoning</h3>
                </div>
                <p className="text-slate-700 leading-relaxed text-lg">
                  {result.reasoning}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
