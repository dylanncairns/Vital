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
  dateBreakRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
    marginTop: 4,
  },
  dateBreakLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#2A2A32",
    marginRight: 10,
  },
  dateBreakText: {
    color: "#A1A1B0",
    fontSize: 11,
    letterSpacing: 0.8,
    textTransform: "uppercase",
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 0,
    paddingBottom: 10,
  },
  treeWrap: {
    width: 34,
    alignItems: "center",
    position: "relative",
    alignSelf: "stretch",
  },
  rail: {
    position: "absolute",
    top: -12,
    bottom: -12,
    width: 2,
    backgroundColor: "#2A2A32",
  },
  nodeWrap: {
    position: "absolute",
    top: 18,
    left: 11,
    flexDirection: "row",
    alignItems: "center",
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  branch: {
    width: 12,
    height: 2,
    backgroundColor: "#2A2A32",
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
