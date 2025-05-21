import { Alert, Camera, CrowdData, DetectionEvent, SecurityStat } from '../types';

// Mock cameras
export const mockCameras: Camera[] = [
  {
    id: 'cam-001',
    name: 'Main Entrance',
    location: 'Building A - Front',
    status: 'online',
    streamUrl: 'https://images.pexels.com/photos/4318822/pexels-photo-4318822.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
  {
    id: 'cam-002',
    name: 'Parking Lot',
    location: 'Southern Perimeter',
    status: 'online',
    streamUrl: 'https://images.pexels.com/photos/1829191/pexels-photo-1829191.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
  {
    id: 'cam-003',
    name: 'Main Lobby',
    location: 'Building A - Ground Floor',
    status: 'online',
    streamUrl: 'https://images.pexels.com/photos/2898285/pexels-photo-2898285.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
  {
    id: 'cam-004',
    name: 'Warehouse Entrance',
    location: 'Building B - East Wing',
    status: 'recording',
    streamUrl: 'https://images.pexels.com/photos/6969952/pexels-photo-6969952.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
  {
    id: 'cam-005',
    name: 'Server Room',
    location: 'Building A - Level 3',
    status: 'offline',
    streamUrl: 'https://images.pexels.com/photos/325229/pexels-photo-325229.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
  {
    id: 'cam-006',
    name: 'Emergency Exit',
    location: 'Building A - West Wing',
    status: 'online',
    streamUrl: 'https://images.pexels.com/photos/3760958/pexels-photo-3760958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  },
];

// Mock alerts
export const mockAlerts: Alert[] = [
  {
    id: 'alert-001',
    cameraId: 'cam-001',
    timestamp: new Date(Date.now() - 1000 * 60 * 2), // 2 minutes ago
    type: 'suspicious_activity',
    severity: 'medium',
    description: 'Unusual loitering detected at main entrance',
    acknowledged: false,
    thumbnail: 'https://images.pexels.com/photos/4318822/pexels-photo-4318822.jpeg?auto=compress&cs=tinysrgb&w=300',
  },
  {
    id: 'alert-002',
    cameraId: 'cam-002',
    timestamp: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
    type: 'weapon',
    severity: 'critical',
    description: 'Potential firearm detected in parking lot',
    acknowledged: true,
    thumbnail: 'https://images.pexels.com/photos/1829191/pexels-photo-1829191.jpeg?auto=compress&cs=tinysrgb&w=300',
  },
  {
    id: 'alert-003',
    cameraId: 'cam-003',
    timestamp: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
    type: 'crowd_density',
    severity: 'high',
    description: 'Crowd density exceeding 85% in main lobby',
    acknowledged: false,
    thumbnail: 'https://images.pexels.com/photos/2898285/pexels-photo-2898285.jpeg?auto=compress&cs=tinysrgb&w=300',
  },
  {
    id: 'alert-004',
    cameraId: 'cam-004',
    timestamp: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
    type: 'suspicious_activity',
    severity: 'low',
    description: 'Unauthorized access attempt at warehouse entrance',
    acknowledged: true,
    thumbnail: 'https://images.pexels.com/photos/6969952/pexels-photo-6969952.jpeg?auto=compress&cs=tinysrgb&w=300',
  },
];

// Mock crowd density data
export const mockCrowdData: CrowdData[] = [
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6), cameraId: 'cam-003', density: 25, location: 'Main Lobby' },
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 5), cameraId: 'cam-003', density: 40, location: 'Main Lobby' },
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4), cameraId: 'cam-003', density: 65, location: 'Main Lobby' },
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 3), cameraId: 'cam-003', density: 85, location: 'Main Lobby' },
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), cameraId: 'cam-003', density: 75, location: 'Main Lobby' },
  { timestamp: new Date(Date.now() - 1000 * 60 * 60 * 1), cameraId: 'cam-003', density: 60, location: 'Main Lobby' },
  { timestamp: new Date(), cameraId: 'cam-003', density: 45, location: 'Main Lobby' },
];

// Mock detection events
export const mockDetectionEvents: DetectionEvent[] = [
  {
    id: 'event-001',
    cameraId: 'cam-001',
    timestamp: new Date(Date.now() - 1000 * 60 * 2), // 2 minutes ago
    type: 'suspicious_activity',
    confidence: 76,
    boundingBox: { x: 220, y: 170, width: 80, height: 160 },
    details: 'Person loitering near restricted area',
  },
  {
    id: 'event-002',
    cameraId: 'cam-002',
    timestamp: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
    type: 'weapon',
    confidence: 92,
    boundingBox: { x: 320, y: 210, width: 60, height: 40 },
    details: 'Handgun detected in waistband',
  },
  {
    id: 'event-003',
    cameraId: 'cam-003',
    timestamp: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
    type: 'crowd_density',
    confidence: 89,
    details: 'Crowd density exceeding threshold',
  },
];

// Mock security statistics
export const mockSecurityStats: SecurityStat[] = [
  { label: 'Total Alerts Today', value: 27, change: 12, icon: 'Bell' },
  { label: 'Weapon Detections', value: 3, change: -25, icon: 'ShieldAlert' },
  { label: 'Suspicious Activities', value: 14, change: 8, icon: 'Eye' },
  { label: 'Avg. Response Time', value: 2.5, change: -18, icon: 'Clock' }, // minutes
];

// Function to format the date for display
export const formatDate = (date: Date): string => {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};