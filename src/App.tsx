// Code and architecture created by Luis Villeda
import React, { useState, useEffect, useRef } from 'react';
import { Car, AlertTriangle, Activity, Info, Upload, Zap, ShieldCheck, ListChecks, Image as ImageIcon, History, X, Download, FileText, ChevronLeft, Save, BookOpen, Camera, Archive, Trash2 } from 'lucide-react';
import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";
import localforage from 'localforage';
import jsPDF from 'jspdf';
import * as htmlToImage from 'html-to-image';

const PROVIDERS = {
  gemini: {
    name: "Gemini",
    models: ["gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash"]
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
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<any>(null);
  const [tempNotes, setTempNotes] = useState("");
  const [showKnowledgeBase, setShowKnowledgeBase] = useState(false);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    localforage.getItem('analysis_history').then((savedHistory) => {
      if (savedHistory) {
        setHistory(savedHistory as any[]);
      }
    }).catch(err => console.error("Error loading history:", err));
  }, []);

  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setImageUrl(url);
      setResult(null); // Clear previous result when new image is uploaded
      return () => URL.revokeObjectURL(url);
    } else {
      setImageUrl(null);
    }
  }, [imageFile]);

  const handleAnalyze = async () => {
    setResult(null); // Clear cache from previous request
    setError(null);
    setIsAnalyzing(true);

    try {
      let persistentImageUrl = null;
      let base64Data = null;
      let mimeType = null;

      if (imageFile) {
        persistentImageUrl = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(imageFile);
        });
        const parts = persistentImageUrl.split(',');
        base64Data = parts[1];
        mimeType = imageFile.type;
      }

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

      let actualModel = selectedModel;
      if (!actualModel.startsWith('gemini')) {
        actualModel = 'gemini-3.1-pro-preview';
      }

      const weatherMap = { 0: 'Clear', 1: 'Rain', 2: 'Snow', 3: 'Fog' };
      const weatherText = weatherMap[weather as keyof typeof weatherMap];

      // Fetch history to enable dynamic learning from past analyst notes
      const savedHistory = await localforage.getItem('analysis_history') as any[] || [];
      const recentNotes = savedHistory
        .filter(h => h.notes && h.notes.trim() !== '')
        .slice(0, 5)
        .map(h => `- Past incident (${h.result.hazard_name}): ${h.notes}`)
        .join('\n');

      const learningContext = recentNotes ? `\n\nHistorical Analyst Feedback (These are notes from PAST analyses. Use these to understand user preferences and evolve your recognition criteria, but DO NOT assume these past incidents are present in the current image):\n${recentNotes}` : '';

      const prompt = `You are an advanced, continuously evolving AI vision system for autonomous driving risk assessment.
      Analyze this driving scenario for risks. You must detect and recognize a wide variety of entities including, but not limited to: vehicles, pedestrians, animals, road debris, traffic signs, infrastructure, unknown objects, and subtle hazards. ONLY report on what is explicitly visible in the current image.

      Telemetry Data:
      - Speed: ${speed} mph
      - Weather Condition: ${weatherText}
      - Time of Day: ${timeOfDay}:00
      - Driver Alertness: ${alertness}
      - Incident Report: ${report}${learningContext}

      Identify all relevant hazards and objects. Provide a highly detailed analysis (at least 2-3 paragraphs of reasoning), image type detection (e.g., Dashcam, Traffic Camera, Mobile Phone, Synthetic), a risk score (0.0 to 1.0), compliance violations, recommendations, and bounding boxes for ALL detected objects (top, left, width, height as percentages like '35%'). 
      
      For bounding boxes, provide a highly specific label (e.g., 'Debris - Tire Tread', 'Pedestrian - Child', 'Unknown Obstacle').
      
      Assign a specific hex color code and a Tailwind color class based on the artifact type:
      - Emergency (Ambulance, Fire Truck, Police, Firefighter, Responder): #ef4444 / bg-red-500
      - Standard Vehicles (Car, SUV, Truck, Bus, Motorcycle): #3b82f6 / bg-blue-500
      - Personnel (Pedestrian, Cyclist, Worker): #22c55e / bg-green-500
      - Hazards/Debris (Debris, Cone, Barrier, Hazard): #eab308 / bg-yellow-500`;

      const parts: any[] = [];
      if (base64Data && mimeType) {
        parts.push({
          inlineData: {
            data: base64Data,
            mimeType: mimeType
          }
        });
      }
      parts.push({ text: prompt });

      const isGemini3 = actualModel.includes('gemini-3');
      const genConfig: any = {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            image_type: { type: Type.STRING, description: "Type of image (e.g., Dashcam, Traffic Camera, Mobile Phone, Synthetic)" },
            hazard_category: { type: Type.INTEGER, description: "1 for low, 2 for medium, 3 for high" },
            hazard_name: { type: Type.STRING },
            risk_score: { type: Type.NUMBER, description: "0.0 to 1.0" },
            reasoning: { type: Type.STRING, description: "Highly detailed reasoning (at least 2-3 paragraphs) explaining the scene, the risks, and the context." },
            compliance: { type: Type.STRING },
            recommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            detections: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  top: { type: Type.STRING, description: "Percentage, e.g., '35%'" },
                  left: { type: Type.STRING, description: "Percentage, e.g., '45%'" },
                  width: { type: Type.STRING, description: "Percentage, e.g., '15%'" },
                  height: { type: Type.STRING, description: "Percentage, e.g., '40%'" },
                  label: { type: Type.STRING },
                  color: { type: Type.STRING, description: "Hex color code for the bounding box based on object type" },
                  colorClass: { type: Type.STRING, description: "Tailwind color class (e.g., bg-red-500, bg-blue-500, bg-green-500, bg-yellow-500)" }
                }
              }
            }
          }
        }
      };

      if (isGemini3) {
        genConfig.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      }

      let response;
      try {
        response = await ai.models.generateContent({
          model: actualModel,
          contents: { parts: parts },
          config: genConfig
        });
      } catch (err: any) {
        if (err?.status === 429 || err?.message?.includes('429') || err?.message?.includes('RESOURCE_EXHAUSTED')) {
          console.warn(`Model ${actualModel} hit rate limit. Falling back to gemini-2.5-flash...`);
          actualModel = 'gemini-2.5-flash';
          if (genConfig.thinkingConfig) {
            delete genConfig.thinkingConfig;
          }
          try {
            response = await ai.models.generateContent({
              model: actualModel,
              contents: { parts: parts },
              config: genConfig
            });
          } catch (err2: any) {
            if (err2?.status === 429 || err2?.message?.includes('429') || err2?.message?.includes('RESOURCE_EXHAUSTED')) {
              console.warn(`Model ${actualModel} hit rate limit. Falling back to gemini-1.5-flash...`);
              actualModel = 'gemini-1.5-flash';
              response = await ai.models.generateContent({
                model: actualModel,
                contents: { parts: parts },
                config: genConfig
              });
            } else {
              throw err2;
            }
          }
        } else {
          throw err;
        }
      }

      const responseText = response.text;
      if (!responseText) throw new Error("No response from AI");
      
      const newResult = JSON.parse(responseText);
      newResult.provider_used = `${PROVIDERS[provider].name} (${selectedModel})`;

      setResult(newResult);
      setHistory(prev => {
        const newHistory = [{
          id: Date.now(),
          date: new Date().toLocaleString(),
          imageName: imageFile ? imageFile.name : 'No Image',
          imageUrl: persistentImageUrl,
          provider: `${PROVIDERS[provider].name} (${selectedModel})`,
          result: newResult,
          notes: ''
        }, ...prev];
        localforage.setItem('analysis_history', newHistory).catch(console.error);
        return newHistory;
      });

    } catch (error: any) {
      console.error("Analysis failed:", error);
      let errorMsg = error.message || "An error occurred during analysis.";
      try {
        // Try to parse JSON error message if it exists
        if (error.message && error.message.startsWith('{')) {
          const parsed = JSON.parse(error.message);
          if (parsed.error && parsed.error.message) {
            errorMsg = parsed.error.message;
          }
        }
      } catch (e) {
        // Ignore parsing errors
      }
      setError(errorMsg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const [isExporting, setIsExporting] = useState(false);

  const handleExportPDF = async (elementId: string, filename: string) => {
    const element = document.getElementById(elementId);
    if (!element) return;

    setIsExporting(true);
    try {
      // Create a clone to avoid disrupting the UI and bypass parent clipping
      const clone = element.cloneNode(true) as HTMLElement;
      const wrapper = document.createElement('div');
      wrapper.style.position = 'absolute';
      wrapper.style.top = '0';
      wrapper.style.left = '0';
      wrapper.style.width = `${element.scrollWidth}px`;
      wrapper.style.opacity = '0';
      wrapper.style.pointerEvents = 'none';
      wrapper.style.zIndex = '-9999';
      
      // Ensure clone is fully expanded
      clone.style.height = 'auto';
      clone.style.overflow = 'visible';
      clone.style.maxHeight = 'none';
      
      wrapper.appendChild(clone);
      document.body.appendChild(wrapper);

      const canvas = await htmlToImage.toCanvas(clone, {
        pixelRatio: 2,
        backgroundColor: '#f8fafc', // match bg-slate-50
      });
      
      // Cleanup clone
      document.body.removeChild(wrapper);
      
      const imgData = canvas.toDataURL('image/jpeg', 1.0);
      const pdf = new jsPDF({
        orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });
      
      pdf.addImage(imgData, 'JPEG', 0, 0, canvas.width, canvas.height);
      pdf.save(filename);
    } catch (error) {
      console.error('Error generating PDF:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeleteHistoryItem = async (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const newHistory = history.filter(item => item.id !== id);
    setHistory(newHistory);
    await localforage.setItem('analysis_history', newHistory);
    if (selectedHistoryItem?.id === id) {
      setSelectedHistoryItem(null);
    }
  };

  const handleArchiveHistoryItem = async (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const newHistory = history.map(item => 
      item.id === id ? { ...item, archived: !item.archived } : item
    );
    setHistory(newHistory);
    await localforage.setItem('analysis_history', newHistory);
    if (selectedHistoryItem?.id === id) {
      setSelectedHistoryItem({ ...selectedHistoryItem, archived: !selectedHistoryItem.archived });
    }
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
              {imageFile ? (
                <span className="text-sm font-medium text-green-700 bg-green-100 px-3 py-1 rounded-md truncate inline-block max-w-full">
                  {imageFile.name}
                </span>
              ) : (
                <span className="text-sm text-slate-500 truncate block px-2">
                  Upload JPG, PNG
                </span>
              )}
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

        {/* Action Buttons */}
        <div className="mt-8 pt-6 border-t border-slate-200 space-y-3">
          <button 
            onClick={() => setShowHistory(true)}
            className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg font-semibold transition-colors"
          >
            <History size={18} />
            Analysis History & Export
          </button>
          <button 
            onClick={() => setShowKnowledgeBase(true)}
            className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg font-semibold transition-colors"
          >
            <BookOpen size={18} />
            Safety Knowledge Base
          </button>
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

          {error && !isAnalyzing && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-6 shadow-sm flex items-start gap-4">
              <div className="bg-red-100 text-red-600 p-2 rounded-lg shrink-0">
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 className="text-lg font-medium text-red-900 mb-1">Analysis Failed</h3>
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            </div>
          )}

          {!result && !isAnalyzing && !error && (
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
              </div>

              {/* Images Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Original Image */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="bg-slate-100 text-slate-600 p-2 rounded-lg">
                      <ImageIcon size={20} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">Original {result.image_type || 'Dashcam'}</h3>
                  </div>
                  <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center">
                    {imageUrl ? (
                      <img src={imageUrl} alt="Original" className="object-cover w-full h-full" />
                    ) : (
                      <span className="text-slate-400 text-sm">No image uploaded</span>
                    )}
                  </div>
                </div>

                {/* Annotated Image */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
                      <Zap size={20} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">AI Risk Detection</h3>
                  </div>
                  <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center cursor-pointer group" onClick={() => setSelectedHistoryItem({ result, imageUrl, provider: result.provider_used, date: new Date().toLocaleString() })}>
                    {imageUrl ? (
                      <>
                        <img src={imageUrl} alt="Annotated" className="object-cover w-full h-full opacity-90" />
                        {result.detections && result.detections.map((box: any, i: number) => (
                          <div 
                            key={i}
                            className="absolute border-[1.5px] rounded-sm transition-all duration-200 hover:border-[3px] hover:z-10 group"
                            style={{ 
                              top: box.top, 
                              left: box.left, 
                              width: box.width, 
                              height: box.height,
                              borderColor: box.color || '#00ff00',
                              backgroundColor: `${box.color || '#00ff00'}33`, // 20% opacity fill
                            }}
                          />
                        ))}
                      </>
                    ) : (
                      <span className="text-slate-400 text-sm">No image to analyze</span>
                    )}
                  </div>
                  {result.detections && result.detections.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {Object.entries(result.detections.reduce((acc: any, curr: any) => {
                        const typeName = curr.label.replace(/^\d+\s*/, '').trim(); 
                        if (!acc[typeName]) {
                          acc[typeName] = { count: 0, colorClass: curr.colorClass || 'bg-blue-500' };
                        }
                        acc[typeName].count += 1;
                        return acc;
                      }, {})).map(([typeName, data]: any, i: number) => (
                        <div key={i} className="flex items-center justify-between bg-white px-3 py-2 rounded-lg border border-slate-200 text-sm font-medium text-slate-700 shadow-sm">
                          <div className="flex items-center gap-3">
                            <div className={`w-3 h-3 rounded-full ${data.colorClass}`}></div>
                            <span>{typeName}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-1 mx-4 border-b border-dotted border-slate-300 h-1 mt-1"></div>
                          <span className={`${data.colorClass} text-white px-2.5 py-0.5 rounded-full text-[11px] font-bold shadow-sm`}>
                            {data.count}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Detailed Analysis Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Reasoning */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm md:col-span-1">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
                      <Info size={20} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">AI Reasoning</h3>
                  </div>
                  <p className="text-slate-700 leading-relaxed text-sm">
                    {result.reasoning}
                  </p>
                </div>

                {/* Compliance */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm md:col-span-1">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="bg-green-100 text-green-600 p-2 rounded-lg">
                      <ShieldCheck size={20} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">Safety & Compliance</h3>
                  </div>
                  <p className="text-slate-700 leading-relaxed text-sm">
                    {result.compliance}
                  </p>
                </div>

                {/* Recommendations */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm md:col-span-1">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="bg-purple-100 text-purple-600 p-2 rounded-lg">
                      <ListChecks size={20} />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">Recommendations</h3>
                  </div>
                  <ul className="list-disc list-inside text-slate-700 text-sm space-y-2">
                    {result.recommendations.map((rec: string, i: number) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Detection Report */}
              <div className="mt-6">
                {/* Detection Summary Report */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">Detection Summary</h3>
                  <div className="space-y-3">
                    {(() => {
                      // 1. Mandatory Data Transformation (reduce)
                      const aggregatedDetections = (result.detections || []).reduce((acc: any, curr: any) => {
                        // Strip out any unique IDs to group by the base label
                        const typeName = curr.label.replace(/^\d+\s*/, '').trim(); 
                        if (!acc[typeName]) {
                          acc[typeName] = { count: 0, colorClass: 'bg-slate-500' };
                        }
                        acc[typeName].count += 1;
                        return acc;
                      }, {});

                      return Object.entries(aggregatedDetections).map(([typeName, data]: any, i: number) => (
                        <div key={i} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">
                          <div className="flex items-center gap-3">
                            <div className={`w-4 h-4 rounded-full ${data.colorClass}`}></div>
                            <span className="font-medium text-slate-700 text-sm">
                              {typeName}
                            </span>
                          </div>
                          <span className="text-slate-400 text-sm flex-1 px-4 overflow-hidden whitespace-nowrap">
                            ....................................................................................................
                          </span>
                          <span className="bg-slate-200 text-slate-800 px-3 py-1 rounded-full text-xs font-bold">
                            {data.count}
                          </span>
                        </div>
                      ));
                    })()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* History Modal */}
      {showHistory && !selectedHistoryItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm transition-opacity p-4 sm:p-6">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
                  <History size={24} />
                </div>
                <h2 className="text-2xl font-bold text-slate-900">Analysis History</h2>
              </div>
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => setShowArchived(!showArchived)} 
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${showArchived ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'}`}
                >
                  <Archive size={16} />
                  {showArchived ? 'View Active' : 'View Archived'}
                </button>
                <button 
                  onClick={() => handleExportPDF('bulk-history-content', 'analysis_history.pdf')} 
                  disabled={isExporting}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                >
                  <Download size={16} />
                  {isExporting ? 'Exporting...' : 'Export Bulk PDF'}
                </button>
                <button onClick={() => setShowHistory(false)} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors">
                  <X size={24} />
                </button>
              </div>
            </div>
            {/* Body */}
            <div id="bulk-history-content" className="flex-1 overflow-y-auto p-6 bg-slate-50">
              {history.filter(item => showArchived ? item.archived : !item.archived).length === 0 ? (
                <div className="text-center py-12 text-slate-500">No {showArchived ? 'archived' : 'active'} analysis history found.</div>
              ) : (
                <div className="space-y-4">
                  {history.filter(item => showArchived ? item.archived : !item.archived).map(item => (
                    <div 
                      key={item.id} 
                      onClick={() => {
                        setSelectedHistoryItem(item);
                        setTempNotes(item.notes || "");
                      }}
                      className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col md:flex-row gap-6 cursor-pointer hover:border-blue-300 hover:shadow-md transition-all"
                    >
                      {/* Thumbnail */}
                      <div className="relative w-full md:w-48 h-32 bg-slate-100 rounded-lg overflow-hidden shrink-0 flex items-center justify-center border border-slate-200">
                        {item.imageUrl ? (
                          <>
                            <img src={item.imageUrl} alt={item.imageName} className="w-full h-full object-cover" />
                            {item.result.boxes && item.result.boxes.map((box: any, i: number) => (
                              <div 
                                key={i}
                                className="absolute border-[1px] rounded-[1px] shadow-[0_0_4px_rgba(0,0,0,0.5)]"
                                style={{ 
                                  top: box.top, 
                                  left: box.left, 
                                  width: box.width, 
                                  height: box.height,
                                  borderColor: box.color || '#00ff00',
                                  boxShadow: `0 0 4px ${box.color || '#00ff00'}80, inset 0 0 4px ${box.color || '#00ff00'}40`
                                }}
                              />
                            ))}
                          </>
                        ) : (
                          <span className="text-slate-400 text-sm">No Image</span>
                        )}
                      </div>
                      {/* Details */}
                      <div className="flex-1 flex flex-col">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <h4 className="font-bold text-slate-900 text-lg">{item.result.hazard_name}</h4>
                            <p className="text-sm text-slate-500">{item.date} • {item.provider}</p>
                          </div>
                          <div className="text-right">
                            <span className={`inline-block px-3 py-1 font-bold rounded-full text-sm ${item.result.risk_score > 0.7 ? 'bg-red-100 text-red-700' : 'bg-orange-100 text-orange-700'}`}>
                              Risk: {item.result.risk_score.toFixed(2)}
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-slate-700 line-clamp-2 mb-4 flex-1">{item.result.reasoning}</p>
                        <div className="flex gap-2 mt-auto">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedHistoryItem(item);
                              setTimeout(() => handleExportPDF('single-analysis-content', `analysis_${item.id}.pdf`), 100);
                            }} 
                            disabled={isExporting}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-md text-sm font-medium transition-colors disabled:opacity-50"
                          >
                            <FileText size={14} />
                            {isExporting ? 'Exporting...' : 'Export PDF'}
                          </button>
                          <div className="flex-1"></div>
                          <button 
                            onClick={(e) => handleArchiveHistoryItem(item.id, e)}
                            className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                            title={item.archived ? "Unarchive" : "Archive"}
                          >
                            <Archive size={16} />
                          </button>
                          <button 
                            onClick={(e) => handleDeleteHistoryItem(item.id, e)}
                            className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {selectedHistoryItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm transition-opacity p-4 sm:p-6">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-300">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => setSelectedHistoryItem(null)} 
                  className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors"
                >
                  <ChevronLeft size={24} />
                </button>
                <h2 className="text-2xl font-bold text-slate-900">Analysis Details</h2>
              </div>
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => handleArchiveHistoryItem(selectedHistoryItem.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${selectedHistoryItem.archived ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 hover:bg-slate-200 text-slate-700'}`}
                >
                  <Archive size={16} />
                  {selectedHistoryItem.archived ? 'Unarchive' : 'Archive'}
                </button>
                <button 
                  onClick={() => handleDeleteHistoryItem(selectedHistoryItem.id)}
                  className="flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-600 rounded-lg text-sm font-medium transition-colors"
                >
                  <Trash2 size={16} />
                  Delete
                </button>
                <button 
                  onClick={() => handleExportPDF('single-analysis-content', `analysis_${selectedHistoryItem.id}.pdf`)} 
                  disabled={isExporting}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                >
                  <Download size={16} />
                  {isExporting ? 'Exporting...' : 'Export PDF'}
                </button>
                <button onClick={() => { setShowHistory(false); setSelectedHistoryItem(null); }} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors">
                  <X size={24} />
                </button>
              </div>
            </div>
            {/* Body */}
            <div id="single-analysis-content" className="flex-1 overflow-y-auto p-6 bg-slate-50 space-y-6">
              {/* Top Section: Image & Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-4">
                  <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Analyzed Image</h3>
                  <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center">
                    {selectedHistoryItem.imageUrl ? (
                      <>
                        <img src={selectedHistoryItem.imageUrl} alt="Dashcam" className="object-cover w-full h-full opacity-90" />
                        {selectedHistoryItem.result.detections && selectedHistoryItem.result.detections.map((box: any, i: number) => (
                          <div 
                            key={i}
                            className="absolute border-[1.5px] rounded-sm transition-all duration-200 hover:border-[3px] hover:z-10 group"
                            style={{ 
                              top: box.top, 
                              left: box.left, 
                              width: box.width, 
                              height: box.height,
                              borderColor: box.color || '#00ff00',
                              backgroundColor: `${box.color || '#00ff00'}33`, // 20% opacity fill
                            }}
                          />
                        ))}
                      </>
                    ) : (
                      <div className="text-slate-400 flex flex-col items-center"><ImageIcon size={48} className="mb-2 opacity-50"/>No Image Provided</div>
                    )}
                  </div>
                  {selectedHistoryItem.result.detections && selectedHistoryItem.result.detections.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {Object.entries(selectedHistoryItem.result.detections.reduce((acc: any, curr: any) => {
                        const typeName = curr.label.replace(/^\d+\s*/, '').trim(); 
                        if (!acc[typeName]) {
                          acc[typeName] = { count: 0, colorClass: curr.colorClass || 'bg-blue-500' };
                        }
                        acc[typeName].count += 1;
                        return acc;
                      }, {})).map(([typeName, data]: any, i: number) => (
                        <div key={i} className="flex items-center justify-between bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 text-sm font-medium text-slate-700 shadow-sm">
                          <div className="flex items-center gap-3">
                            <div className={`w-3 h-3 rounded-full ${data.colorClass}`}></div>
                            <span>{typeName}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-1 mx-4 border-b border-dotted border-slate-300 h-1 mt-1"></div>
                          <span className={`${data.colorClass} text-white px-2.5 py-0.5 rounded-full text-[11px] font-bold shadow-sm`}>
                            {data.count}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div className="space-y-4">
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Hazard Category</h3>
                    <p className="text-2xl font-bold text-slate-900">{selectedHistoryItem.result.hazard_name}</p>
                  </div>
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Risk Score</h3>
                    <p className={`text-2xl font-bold ${selectedHistoryItem.result.risk_score > 0.7 ? 'text-red-600' : 'text-orange-600'}`}>
                      {selectedHistoryItem.result.risk_score.toFixed(2)} <span className="text-sm text-slate-500 font-normal">/ 1.00</span>
                    </p>
                  </div>
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Image Type</h3>
                    <p className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                      <Camera size={18} className="text-slate-400"/>
                      {selectedHistoryItem.result.image_type || 'Dashcam'}
                    </p>
                  </div>
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Provider & Date</h3>
                    <p className="text-lg font-semibold text-slate-900">{selectedHistoryItem.provider}</p>
                    <p className="text-sm text-slate-500">{selectedHistoryItem.date}</p>
                  </div>
                </div>
              </div>

              {/* Detailed Text Sections */}
              <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2 mb-2"><Info size={18} className="text-blue-600"/> AI Reasoning</h3>
                  <p className="text-slate-700 text-sm leading-relaxed">{selectedHistoryItem.result.reasoning}</p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2 mb-2"><ShieldCheck size={18} className="text-green-600"/> Safety & Compliance</h3>
                  <p className="text-slate-700 text-sm leading-relaxed">{selectedHistoryItem.result.compliance}</p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2 mb-2"><ListChecks size={18} className="text-purple-600"/> Recommendations</h3>
                  <ul className="list-disc list-inside text-slate-700 text-sm space-y-1">
                    {selectedHistoryItem.result.recommendations.map((rec: string, i: number) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Notes Section */}
              <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Analyst Notes</h3>
                <textarea 
                  value={tempNotes}
                  onChange={(e) => setTempNotes(e.target.value)}
                  placeholder="Add your investigation notes, follow-up actions, or tags here..."
                  className="w-full border border-slate-300 rounded-lg p-3 text-sm min-h-[100px] focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-y mb-3"
                />
                <div className="flex justify-end gap-3">
                  <button 
                    onClick={() => setSelectedHistoryItem(null)}
                    className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
                  >
                    Close
                  </button>
                  <button 
                    onClick={() => {
                      setHistory(prev => {
                        const newHistory = prev.map(h => h.id === selectedHistoryItem.id ? { ...h, notes: tempNotes } : h);
                        localforage.setItem('analysis_history', newHistory).catch(console.error);
                        return newHistory;
                      });
                      setSelectedHistoryItem({ ...selectedHistoryItem, notes: tempNotes });
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    <Save size={16} />
                    Save Notes
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Knowledge Base Modal */}
      {showKnowledgeBase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm transition-opacity p-4 sm:p-6">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <div className="bg-blue-100 text-blue-600 p-2 rounded-lg">
                  <BookOpen size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">Safety Knowledge Base</h2>
                  <p className="text-sm text-slate-500">Reference guidelines and best practices</p>
                </div>
              </div>
              <button onClick={() => setShowKnowledgeBase(false)} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors">
                <X size={24} />
              </button>
            </div>
            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50">
              <div className="max-w-3xl mx-auto bg-white p-8 rounded-xl border border-slate-200 shadow-sm space-y-8 text-slate-700">
                
                <div className="border-b border-slate-200 pb-6">
                  <h1 className="text-3xl font-bold text-slate-900 mb-4">Reducing the Illegal Passing of School Buses</h1>
                  <h2 className="text-xl font-semibold text-blue-600">Best Practices Guide</h2>
                </div>

                <section className="space-y-4">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2"><Info size={20} className="text-blue-500"/> About & Purpose</h3>
                  <p className="leading-relaxed">
                    Students entering or exiting a school bus on the roadway experience risk from motorists nearby, particularly at school bus stops where students may need to cross the road. Since the earliest school buses hit the road over a century ago, the risks posed by other motorists have been increasingly addressed by the development and deployment of various technologies (e.g., stop-arms) to prevent illegal passes.
                  </p>
                  <p className="leading-relaxed">
                    Today, all states have laws requiring motorists to stop for a stopped school bus with its red lights flashing and stop signal arm deployed. Despite these advancements, school buses are still illegally passed at an alarmingly high rate. Based on a survey of bus drivers by the National Association of State Directors of Pupil Transportation Services (NASDPTS) in 2023, NASDPTS estimates that there were more than <strong>43.5 million illegal school bus passings</strong> in the United States during the 2022-2023 school year.
                  </p>
                </section>

                <section className="space-y-4">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2"><ShieldCheck size={20} className="text-green-500"/> Safe System Approach</h3>
                  <p className="leading-relaxed">The U.S. DOT utilizes the Safe System Approach to characterize and address a variety of roadway safety issues. Core principles include:</p>
                  <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Deaths & Injuries Are Unacceptable:</strong> Prioritizes the elimination of crashes that result in death and serious injuries.</li>
                    <li><strong>Humans Make Mistakes:</strong> The transportation system can be designed and operated to accommodate certain types and levels of human mistakes.</li>
                    <li><strong>Humans Are Vulnerable:</strong> It is critical to design and operate a transportation system that is human-centric and accommodates physical human vulnerabilities.</li>
                    <li><strong>Responsibility Is Shared:</strong> All stakeholders are vital to preventing fatalities and serious injuries on our roadways.</li>
                    <li><strong>Safety Is Proactive:</strong> Proactive tools should be used to identify and address safety issues in the transportation system.</li>
                    <li><strong>Redundancy Is Crucial:</strong> Reducing risks requires that all parts of the transportation system be strengthened.</li>
                  </ul>
                </section>

                <section className="space-y-4">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2"><AlertTriangle size={20} className="text-orange-500"/> The Problem</h3>
                  <p className="leading-relaxed">
                    The act of illegally passing a stopped school bus with red lights flashing is commonly known as a "stop-arm violation." A recent national survey asked over 3,500 people why they think people illegally pass a stopped school bus. The top four reported reasons were that violators:
                  </p>
                  <ul className="list-disc pl-6 space-y-1">
                    <li>Didn’t care (30.5%)</li>
                    <li>Were in a hurry (25.5%)</li>
                    <li>Didn’t know the law (24.3%)</li>
                    <li>Were distracted (12.2%)</li>
                  </ul>
                  <p className="leading-relaxed mt-4">
                    <strong>The Law:</strong> Every one of the 50 states, the District of Columbia, Guam, Puerto Rico, and the U.S. Virgin Islands has a law making it illegal to pass a school bus with its red lights flashing and stop-arm deployed indicating it is stopped to load or unload students.
                  </p>
                </section>

                <section className="space-y-4">
                  <h3 className="text-2xl font-bold text-slate-900 flex items-center gap-2"><ListChecks size={20} className="text-purple-500"/> Developing a Program</h3>
                  <p className="leading-relaxed">Most stop-arm compliance programs include four basic components:</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <div className="bg-slate-100 p-4 rounded-lg">
                      <h4 className="font-bold text-slate-900 mb-2">1. Enforcement</h4>
                      <p className="text-sm">Activities designed to increase compliance with laws governing the passing of school buses, including routine and selective enforcement (e.g., hot spots, automated camera systems).</p>
                    </div>
                    <div className="bg-slate-100 p-4 rounded-lg">
                      <h4 className="font-bold text-slate-900 mb-2">2. Engineering</h4>
                      <p className="text-sm">Issues such as road design, construction, and signage elements intended to decrease rates of stop-arm violations or otherwise enhance safety for school bus passengers.</p>
                    </div>
                    <div className="bg-slate-100 p-4 rounded-lg">
                      <h4 className="font-bold text-slate-900 mb-2">3. Education & Awareness</h4>
                      <p className="text-sm">Public awareness campaigns educating parents, students, teachers, administrators, or the public to increase their knowledge of the dangers of this situation.</p>
                    </div>
                    <div className="bg-slate-100 p-4 rounded-lg">
                      <h4 className="font-bold text-slate-900 mb-2">4. Policy / Legislation</h4>
                      <p className="text-sm">Activities aimed at getting legislation passed or policies established at the state or local level to increase student transportation safety (e.g., fines, points).</p>
                    </div>
                  </div>
                </section>

              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
