export interface Camera {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'recording';
  streamUrl: string; // In a real app, this would be the actual video stream URL
}

export interface Alert {
  id: string;
  cameraId: string;
  timestamp: Date;
  type: 'weapon' | 'suspicious_activity' | 'crowd_density';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  acknowledged: boolean;
  thumbnail: string; // URL to event thumbnail
}

export interface CrowdData {
  timestamp: Date;
  cameraId: string;
  density: number; // 0-100 percentage
  location: string;
}

export interface DetectionEvent {
  id: string;
  cameraId: string;
  timestamp: Date;
  type: 'weapon' | 'suspicious_activity' | 'crowd_density';
  confidence: number; // 0-100 percentage
  boundingBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  details: string;
}

export interface SecurityStat {
  label: string;
  value: number;
  change: number; // Percentage change from previous period
  icon: string;
}