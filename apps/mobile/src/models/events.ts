export type TimelineEvent = {
  id: number;
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp: string;
  item_id?: number | null;
  item_name?: string | null;
  route?: string | null;
  symptom_id?: number | null;
  symptom_name?: string | null;
  severity?: number | null;
};

export type CreateEventRequest = {
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp: string;
  item_id?: number | null;
  route?: string | null;
  symptom_id?: number | null;
  severity?: number | null;
};