import React, { useState } from 'react';
import { Camera as CameraIcon, Maximize2, Pause, Play, RefreshCw } from 'lucide-react';
import { Camera } from '../../types';

interface VideoFeedProps {
  camera: Camera;
  onMaximize?: (camera: Camera) => void;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ camera, onMaximize }) => {
  const [isPaused, setIsPaused] = useState(false);

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden h-full flex flex-col shadow-md transform transition-all hover:shadow-lg">
      <div className="relative">
        {/* Video thumbnail/stream */}
        <div className="aspect-video w-full overflow-hidden bg-gray-800 relative">
          <img 
            src={camera.streamUrl} 
            alt={`Feed from ${camera.name}`} 
            className={`w-full h-full object-cover ${isPaused ? 'opacity-60' : 'opacity-100'}`}
          />
          
          {isPaused && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-gray-900 bg-opacity-70 rounded-full p-3">
                <Pause className="h-8 w-8 text-white" />
              </div>
            </div>
          )}
          
          {/* Camera status indicator */}
          <div className="absolute top-3 left-3 flex items-center">
            <div className={`h-2.5 w-2.5 rounded-full mr-1.5 ${
              camera.status === 'online' ? 'bg-green-500' : 
              camera.status === 'recording' ? 'bg-red-500 animate-pulse' : 
              'bg-gray-500'
            }`}></div>
            <span className="text-xs font-medium text-white bg-black bg-opacity-50 py-1 px-2 rounded-md">
              {camera.status.charAt(0).toUpperCase() + camera.status.slice(1)}
            </span>
          </div>
          
          {/* Camera name */}
          <div className="absolute bottom-3 left-3 right-3 flex justify-between items-center">
            <div className="flex items-center bg-black bg-opacity-50 py-1 px-2 rounded-md">
              <CameraIcon className="h-3.5 w-3.5 text-white mr-1.5" />
              <span className="text-xs font-medium text-white">{camera.name}</span>
            </div>
            <div className="flex space-x-1">
              <button 
                onClick={() => setIsPaused(!isPaused)}
                className="p-1.5 bg-black bg-opacity-50 rounded-full hover:bg-opacity-70 transition-colors"
                aria-label={isPaused ? "Play" : "Pause"}
              >
                {isPaused ? 
                  <Play className="h-3.5 w-3.5 text-white" /> : 
                  <Pause className="h-3.5 w-3.5 text-white" />
                }
              </button>
              <button 
                onClick={() => onMaximize && onMaximize(camera)}
                className="p-1.5 bg-black bg-opacity-50 rounded-full hover:bg-opacity-70 transition-colors"
                aria-label="Fullscreen"
              >
                <Maximize2 className="h-3.5 w-3.5 text-white" />
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div className="p-3 bg-gray-850 flex-1">
        <div className="flex justify-between items-center">
          <div>
            <p className="text-xs text-gray-400">{camera.location}</p>
          </div>
          <button 
            className="text-gray-400 hover:text-blue-400 transition-colors"
            aria-label="Refresh feed"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoFeed;