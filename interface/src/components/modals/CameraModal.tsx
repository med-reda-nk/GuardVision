import React from 'react';
import { X, Download, Flag, Play, Pause } from 'lucide-react';
import { Camera } from '../../types';

interface CameraModalProps {
  camera: Camera;
  onClose: () => void;
}

const CameraModal: React.FC<CameraModalProps> = ({ camera, onClose }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-75 p-4">
      <div className="bg-gray-900 rounded-lg shadow-2xl max-w-5xl w-full max-h-[90vh] flex flex-col overflow-hidden">
        <div className="p-4 border-b border-gray-800 flex justify-between items-center">
          <h2 className="text-white font-semibold flex items-center">
            <span className={`h-2 w-2 rounded-full mr-2 ${
              camera.status === 'online' ? 'bg-green-500' : 
              camera.status === 'recording' ? 'bg-red-500' : 
              'bg-gray-500'
            }`}></span>
            {camera.name} - {camera.location}
          </h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <div className="flex-1 overflow-hidden relative">
          <img 
            src={camera.streamUrl} 
            alt={`Feed from ${camera.name}`} 
            className="w-full h-full object-contain"
          />
          
          {/* AI Detection overlay - simulating weapon detection */}
          <div className="absolute top-[30%] left-[40%] w-[60px] h-[40px] border-2 border-red-500 rounded-sm flex items-center justify-center animate-pulse opacity-70">
            <span className="text-xs font-bold text-red-500 bg-black bg-opacity-50 px-1">WEAPON</span>
          </div>
          
          {/* AI Detection overlay - simulating person detection */}
          <div className="absolute top-[20%] left-[38%] w-[80px] h-[180px] border-2 border-blue-400 rounded-sm flex items-start justify-center opacity-70">
            <span className="text-xs font-bold text-blue-400 bg-black bg-opacity-50 px-1 -mt-5">PERSON</span>
          </div>
        </div>
        
        <div className="p-4 bg-gray-800 flex items-center justify-between">
          <div className="flex space-x-3">
            <button className="p-2 bg-blue-700 hover:bg-blue-800 rounded-full text-white transition-colors">
              <Play className="h-5 w-5" />
            </button>
            <button className="p-2 bg-gray-700 hover:bg-gray-600 rounded-full text-white transition-colors">
              <Pause className="h-5 w-5" />
            </button>
          </div>
          
          <div className="flex space-x-4">
            <button className="flex items-center text-gray-300 hover:text-white transition-colors">
              <Download className="h-5 w-5 mr-1.5" />
              <span className="text-sm">Export</span>
            </button>
            <button className="flex items-center text-gray-300 hover:text-white transition-colors">
              <Flag className="h-5 w-5 mr-1.5" />
              <span className="text-sm">Report</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraModal;