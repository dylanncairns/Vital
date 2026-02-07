import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#0E0E12",
  },
  container: {
    padding: 20,
    paddingBottom: 40,
    backgroundColor: "#0E0E12",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 20,
  },
  headerTitle: {
    color: "#E6E6E8",
    fontSize: 24,
    fontWeight: "700",
  },
  headerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#2A2A32",
    marginHorizontal: 12,
  },
  headerDate: {
    color: "#8A8A9A",
    fontSize: 12,
    letterSpacing: 1,
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 18,
  },
  railWrap: {
    width: 30,
    alignItems: "center",
    position: "relative",
  },
  rail: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: "#2A2A32",
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: 18,
  },
  card: {
    flex: 1,
    backgroundColor: "#1A1A22",
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: "#2A2A32",
  },
  timeText: {
    color: "#A0A0B2",
    fontSize: 12,
    marginBottom: 6,
  },
  titleText: {
    color: "#F0F0F5",
    fontSize: 16,
    fontWeight: "700",
    marginBottom: 6,
  },
  bodyText: {
    color: "#B6B6C6",
    fontSize: 13,
    marginBottom: 6,
  },
  metaText: {
    color: "#7E7E92",
    fontSize: 11,
  },
});